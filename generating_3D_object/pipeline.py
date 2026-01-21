import os
import time
import yaml
import logging
import httpx
from pathlib import Path
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from tripo import Client
import json
from datetime import datetime
from threading import Semaphore, Lock
import queue
import shutil
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class ModelGenerationResult:
    image_path: str
    task_id: Optional[str]
    output_path: Optional[str]
    preview_path: Optional[str]
    status: str  # 'success', 'failed', 'pending'
    error_message: Optional[str] = None
    processing_time: Optional[float] = None
    model_url: Optional[str] = None
    preview_url: Optional[str] = None
    folder_path: Optional[str] = None


class RateLimiter:
    def __init__(self, max_calls: int, time_window: float = 1.0):
        """
        Args:
            max_calls: Maximum calls allowed in time window
            time_window: Time window in seconds
        """
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = queue.Queue()
        self.lock = Lock()

    def acquire(self):
        with self.lock:
            now = time.time()

            # Remove old calls outside the time window
            while not self.calls.empty():
                call_time = self.calls.queue[0]
                if now - call_time > self.time_window:
                    self.calls.get()
                else:
                    break

            # If at limit, wait
            if self.calls.qsize() >= self.max_calls:
                oldest_call = self.calls.queue[0]
                wait_time = self.time_window - (now - oldest_call)
                if wait_time > 0:
                    logger.debug(f"Rate limit reached, waiting {wait_time:.2f}s")
                    time.sleep(wait_time)
                    return self.acquire()  # Retry after waiting

            # Record this call
            self.calls.put(now)


class TripoModelGenerator:
    def __init__(self, config_path: str = 'config.yaml'):
        self.config = self._load_config(config_path)
        self.results: List[ModelGenerationResult] = []

        self.upload_limiter = RateLimiter(
            max_calls=self.config['tripo']['limits']['upload_qps'],
            time_window=1.0
        )

        self.task_semaphore = Semaphore(
            self.config['tripo']['limits']['concurrent_tasks']
        )

    def _load_config(self, config_path: str) -> Dict:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def _create_client(self) -> Client:
        api_key = os.getenv("TRIPO_API_KEY")
        if not api_key:
            raise RuntimeError("TRIPO_API_KEY not set in .env file")

        client = Client(api_key=api_key)

        # Set extended timeouts to prevent upload failures
        timeout = httpx.Timeout(
            connect=10.0,
            read=120.0,
            write=120.0,
            pool=None,
        )

        client.client = httpx.Client(
            base_url=client.base_url,
            headers=client.headers,
            timeout=timeout,
        )

        return client

    def _get_image_type(self, image_path: str) -> str:
        ext = Path(image_path).suffix.lower()
        if ext in ['.jpg', '.jpeg']:
            return 'jpg'
        elif ext == '.png':
            return 'png'
        else:
            logger.warning(f"Unknown image type {ext}, defaulting to jpg")
            return 'jpg'

    def generate_single_model(self, image_path: str, output_dir: Path) -> ModelGenerationResult:
        start_time = time.time()
        image_name = Path(image_path).stem

        # Create a dedicated folder for this model
        model_folder = output_dir / image_name
        model_folder.mkdir(parents=True, exist_ok=True)

        # Define output paths within the model folder
        output_path = model_folder / f"{image_name}.glb"
        preview_path = model_folder / f"{image_name}_preview.png"
        original_image_path = model_folder / f"{image_name}{Path(image_path).suffix}"

        logger.info(f"Processing: {image_path}")
        logger.info(f"Output folder: {model_folder}")

        self.task_semaphore.acquire()

        client = None
        try:
            client = self._create_client()
            try:
                balance = client.get_balance()
                logger.debug(f"Balance: {balance.balance}, Frozen: {balance.frozen}")
            except Exception as e:
                logger.warning(f"Could not fetch balance: {e}")

            logger.info(f"Uploading: {image_path}")
            self.upload_limiter.acquire()  # Respect QPS limit
            file_token = client.upload_file(image_path)
            logger.info(f"File token: {file_token.file_token}")

            # Prepare advanced generation parameters via direct API call
            payload = {
                "type": "image_to_model",
                "model_version": self.config['tripo']['model_version'],
                "file": {
                    "type": self._get_image_type(image_path),
                    "file_token": file_token.file_token
                },
                "pbr": self.config['tripo'].get('pbr', True),
            }

            model_version = self.config['tripo']['model_version']

            # Parameters for v2.0+ models
            if 'v2.' in model_version or 'v3.' in model_version or 'Turbo' in model_version:
                # Texture alignment for matching original image
                if self.config['tripo'].get('texture_alignment'):
                    payload['texture_alignment'] = self.config['tripo']['texture_alignment']

                # Texture quality
                if self.config['tripo'].get('texture_quality'):
                    payload['texture_quality'] = self.config['tripo']['texture_quality']

                # Orientation to match image angle
                if self.config['tripo'].get('orientation'):
                    payload['orientation'] = self.config['tripo']['orientation']

                # Auto-size control
                if 'auto_size' in self.config['tripo']:
                    payload['auto_size'] = self.config['tripo']['auto_size']

                # Model and texture seeds for reproducibility
                if self.config['tripo'].get('model_seed'):
                    payload['model_seed'] = self.config['tripo']['model_seed']
                if self.config['tripo'].get('texture_seed'):
                    payload['texture_seed'] = self.config['tripo']['texture_seed']

                # Face limit
                if self.config['tripo'].get('face_limit'):
                    payload['face_limit'] = self.config['tripo']['face_limit']

            # Parameters for v3.0+ models
            if 'v3.' in model_version:
                if self.config['tripo'].get('geometry_quality'):
                    payload['geometry_quality'] = self.config['tripo']['geometry_quality']

            logger.info(f"Creating task with advanced parameters for {image_name}")
            response = client.client.post(
                "https://api.tripo3d.ai/v2/openapi/task",
                json=payload
            )
            response.raise_for_status()
            result = response.json()

            if result.get("code") != 0:
                raise RuntimeError(f"API error: {result}")

            task_id = result["data"]["task_id"]
            logger.info(f"Task created: {task_id} for {image_name}")

            max_wait = self.config['processing']['max_wait_time']
            poll_interval = self.config['processing']['poll_interval']
            elapsed = 0
            attempt = 0

            while elapsed < max_wait:
                attempt += 1
                task_status = client.get_task(task_id)
                logger.debug(f"[attempt {attempt}] {image_name} status = {task_status.status}")

                if task_status.status == "success":
                    # Extract model URLs
                    out = task_status.output
                    logger.debug(f"Task output - base_model: {getattr(out, 'base_model', None)}")
                    logger.debug(f"Task output - model: {getattr(out, 'model', None)}")
                    logger.debug(f"Task output - pbr_model: {getattr(out, 'pbr_model', None)}")

                    # Pick best available model URL
                    model_url = (
                            getattr(out, "pbr_model", None)
                            or getattr(out, "model", None)
                            or getattr(out, "base_model", None)
                    )

                    if not model_url:
                        raise RuntimeError(f"Task {task_id} success but no model URL")

                    # Download GLB model
                    logger.info(f"Downloading model from: {model_url}")
                    glb_bytes = client._download_model(model_url)
                    with open(output_path, "wb") as f:
                        f.write(glb_bytes)
                    logger.info(f"Saved: {output_path}")

                    # Download preview if available
                    preview_url = getattr(out, "rendered_image", None)
                    saved_preview = None
                    if preview_url:
                        logger.info(f"Downloading preview from: {preview_url}")
                        png_bytes = client._download_model(preview_url)
                        with open(preview_path, "wb") as f:
                            f.write(png_bytes)
                        logger.info(f"Saved preview: {preview_path}")
                        saved_preview = str(preview_path)

                    # Copy original image to the model folder
                    shutil.copy2(image_path, original_image_path)
                    logger.info(f"Copied original image: {original_image_path}")

                    processing_time = time.time() - start_time
                    logger.info(f"Completed: {image_name} ({processing_time:.2f}s)")

                    return ModelGenerationResult(
                        image_path=image_path,
                        task_id=task_id,
                        output_path=str(output_path),
                        preview_path=saved_preview,
                        status='success',
                        processing_time=processing_time,
                        model_url=model_url,
                        preview_url=preview_url,
                        folder_path=str(model_folder)
                    )

                elif task_status.status in ("failed", "canceled"):
                    error_msg = f"Task ended with status: {task_status.status}"
                    logger.error(f"Failed: {image_name} - {error_msg}")
                    return ModelGenerationResult(
                        image_path=image_path,
                        task_id=task_id,
                        output_path=None,
                        preview_path=None,
                        status='failed',
                        error_message=error_msg,
                        folder_path=str(model_folder)
                    )

                elif task_status.status in ("pending", "running"):
                    time.sleep(poll_interval)
                    elapsed += poll_interval
                else:
                    logger.warning(f"Unknown status: {task_status.status}")
                    time.sleep(poll_interval)
                    elapsed += poll_interval

            # Timeout
            error_msg = f"Timeout after {max_wait}s"
            logger.error(f"Failed: {image_name} - {error_msg}")
            return ModelGenerationResult(
                image_path=image_path,
                task_id=task_id,
                output_path=None,
                preview_path=None,
                status='failed',
                error_message=error_msg,
                folder_path=str(model_folder)
            )

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Error processing {image_name}: {str(e)}")
            return ModelGenerationResult(
                image_path=image_path,
                task_id=None,
                output_path=None,
                preview_path=None,
                status='failed',
                error_message=str(e),
                processing_time=processing_time,
                folder_path=str(model_folder) if model_folder.exists() else None
            )
        finally:
            if client:
                try:
                    client.close()
                except:
                    pass
            self.task_semaphore.release()

    def process_batch(self, image_paths: List[str], output_dir: Path) -> List[ModelGenerationResult]:
        output_dir.mkdir(parents=True, exist_ok=True)

        max_workers = self.config['processing']['max_workers']
        logger.info(f"Starting batch processing: {len(image_paths)} images")
        logger.info(f"Model version: {self.config['tripo']['model_version']}")
        logger.info(f"Thread pool: {max_workers} workers")
        logger.info(f"Concurrent tasks limit: {self.config['tripo']['limits']['concurrent_tasks']}")
        logger.info(f"Upload rate limit: {self.config['tripo']['limits']['upload_qps']} QPS")

        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_image = {
                executor.submit(self.generate_single_model, img, output_dir): img
                for img in image_paths
            }

            # Collect results as they complete
            for future in as_completed(future_to_image):
                result = future.result()
                results.append(result)

                # Progress update
                completed = len(results)
                total = len(image_paths)
                success = sum(1 for r in results if r.status == 'success')
                failed = sum(1 for r in results if r.status == 'failed')
                logger.info(f"Progress: {completed}/{total} | Success {success} | Failed {failed}")

        self.results = results
        return results

    def save_report(self, output_dir: Path):
        total_time = sum(r.processing_time for r in self.results if r.processing_time)
        avg_time = total_time / len(self.results) if self.results else 0

        report = {
            'timestamp': datetime.now().isoformat(),
            'total_images': len(self.results),
            'successful': sum(1 for r in self.results if r.status == 'success'),
            'failed': sum(1 for r in self.results if r.status == 'failed'),
            'total_processing_time': round(total_time, 2),
            'average_processing_time': round(avg_time, 2),
            'config': {
                'model_version': self.config['tripo']['model_version'],
                'max_workers': self.config['processing']['max_workers'],
                'concurrent_tasks_limit': self.config['tripo']['limits']['concurrent_tasks'],
                'upload_qps_limit': self.config['tripo']['limits']['upload_qps'],
                'advanced_params': {
                    'texture_alignment': self.config['tripo'].get('texture_alignment'),
                    'texture_quality': self.config['tripo'].get('texture_quality'),
                    'geometry_quality': self.config['tripo'].get('geometry_quality'),
                    'orientation': self.config['tripo'].get('orientation'),
                }
            },
            'results': [
                {
                    'image': r.image_path,
                    'task_id': r.task_id,
                    'folder': r.folder_path,
                    'output': r.output_path,
                    'preview': r.preview_path,
                    'status': r.status,
                    'error': r.error_message,
                    'processing_time': r.processing_time,
                    'model_url': r.model_url,
                    'preview_url': r.preview_url
                }
                for r in self.results
            ]
        }

        report_path = output_dir / 'processing_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"Report saved: {report_path}")
        return report

    def retry_failed(self, output_dir: Path) -> List[ModelGenerationResult]:
        report_path = output_dir / 'processing_report.json'

        if not report_path.exists():
            logger.warning("No previous report found")
            return []

        with open(report_path, 'r') as f:
            report = json.load(f)

        failed_images = [
            r['image'] for r in report['results']
            if r['status'] == 'failed'
        ]

        if not failed_images:
            logger.info("No failed images to retry")
            return []

        logger.info(f"Retrying {len(failed_images)} failed images")
        return self.process_batch(failed_images, output_dir)


def discover_images(input_dir: Path, extensions: List[str]) -> List[str]:
    images = []
    for ext in extensions:
        images.extend(input_dir.glob(f"*.{ext}"))
        images.extend(input_dir.glob(f"*.{ext.upper()}"))
    return [str(img) for img in sorted(images)]


def main():
    # Load config relative to this script
    script_dir = Path(__file__).parent
    config_path = script_dir / 'config.yaml'
    generator = TripoModelGenerator(str(config_path))
    config = generator.config

    # Setup paths relative to project root
    project_root = script_dir.parent
    input_dir = project_root / config['paths']['input_dir']
    output_dir = project_root / config['paths']['output_dir']

    # Discover images
    image_paths = discover_images(input_dir, config['paths']['image_extensions'])
    logger.info(f"Found {len(image_paths)} images in {input_dir}")

    if not image_paths:
        logger.error("No images found!")
        return

    # Process batch
    results = generator.process_batch(image_paths, output_dir)

    report = generator.save_report(output_dir)

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETED")
    logger.info(f"Total: {report['total_images']}")
    logger.info(f"Success: {report['successful']}")
    logger.info(f"Failed: {report['failed']}")
    logger.info(f"Total Time: {report['total_processing_time']:.2f}s")
    logger.info(f"Avg Time: {report['average_processing_time']:.2f}s per model")


if __name__ == "__main__":
    main()
