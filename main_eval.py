import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

# Local imports
import model_discovery
import metrics_image
import metrics_mesh
import score_calculator

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s"
)
logger = logging.getLogger(__name__)

LPIPS_MODEL = None
CLIP_MODEL = None
CLIP_PREPROCESS = None


def load_ai_models():
    global LPIPS_MODEL, CLIP_MODEL, CLIP_PREPROCESS
    
    try:
        import lpips
        LPIPS_MODEL = lpips.LPIPS(net='alex')
        logger.info("  LPIPS loaded")
    except ImportError:
        logger.warning("  LPIPS library not found (optional)")
    except Exception as e:
        logger.warning(f"  LPIPS failed to load: {e}")

    try:
        import torch
        try:
            import open_clip
            model, _, preprocess = open_clip.create_model_and_transforms(
                'ViT-B-32', pretrained='laion2b_s34b_b79k'
            )
            logger.info("  CLIP loaded (OpenCLIP)")
            CLIP_MODEL = model
            CLIP_PREPROCESS = preprocess
        except ImportError:
            import clip
            model, preprocess = clip.load("ViT-B/32")
            logger.info("  CLIP loaded (OpenAI)")
            CLIP_MODEL = model
            CLIP_PREPROCESS = preprocess

    except ImportError:
        logger.warning("  CLIP/Torch libraries not found (optional)")
    except Exception as e:
        logger.warning(f"  CLIP failed to load: {e}")


def evaluate_single_model(model_info: Dict) -> Dict:
    result = {
        'name': model_info['name'],
        'original_image': str(model_info['image']),
        'preview_image': str(model_info['preview']),
        'model': str(model_info['glb']),
        'folder': str(model_info['folder']),
        'status': 'pending',
        'metrics': {},
        'shape_score': 0.0,
        'appearance_score': 0.0,
        'semantic_score': 0.0,
        'quality_score': 0.0,
    }

    try:
        input_image = str(model_info['image'])
        preview_image = str(model_info['preview'])

        # Pass loaded models into metrics_image
        metrics = metrics_image.calculate_basic_metrics(
            input_image, 
            preview_image,
            lpips_model=LPIPS_MODEL,
            clip_model=CLIP_MODEL,
            clip_preprocess=CLIP_PREPROCESS
        )
        result['metrics'] = metrics

        glb_path = str(model_info['glb'])
        mesh_metrics = metrics_mesh.analyze_model(glb_path)
        result['metrics'].update(mesh_metrics)

        scores = score_calculator.calculate_scores(result['metrics'])
        result.update(scores)

        result['status'] = 'success'
        logger.info(
            f"  Scores -> Quality: {result['quality_score']:.1f} | "
            f"Shape: {result['shape_score']:.0f} | "
            f"Appearance: {result['appearance_score']:.0f} | "
            f"Semantic: {result['semantic_score']:.0f}"
        )

    except Exception as e:
        result['status'] = 'failed'
        result['error'] = str(e)
        logger.error(f"  Evaluation failed: {e}")

    return result


def save_reports(results: List[Dict], output_dir: Path):
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] == 'failed']
    quality_scores = [r['quality_score'] for r in successful]

    has_lpips = LPIPS_MODEL is not None
    has_clip = CLIP_MODEL is not None
    try:
        import trimesh
        has_trimesh = True
    except ImportError:
        has_trimesh = False
    try:
        import skimage
        has_ssim = True
    except ImportError:
        has_ssim = False

    summary = {
        'total': len(results),
        'successful': len(successful),
        'failed': len(failed),
        'statistics': {
            'mean_quality': round(sum(quality_scores) / len(quality_scores), 2) if quality_scores else 0,
            'min_quality': round(min(quality_scores), 2) if quality_scores else 0,
            'max_quality': round(max(quality_scores), 2) if quality_scores else 0,
        },
        'available_metrics': {
            'ssim': has_ssim,
            'lpips': has_lpips,
            'semantic': has_clip,
            'mesh_analysis': has_trimesh,
        },
        'results': results
    }

    report_path = output_dir / 'evaluation_report.json'
    with open(report_path, 'w') as f:
        json.dump(summary, f, indent=2)

    ranked = sorted(successful, key=lambda x: x['quality_score'], reverse=True)
    rankings = {
        'models': [
            {
                'rank': i + 1,
                'name': r['name'],
                'quality': r['quality_score'],
                'shape': r['shape_score'],
                'appearance': r['appearance_score'],
                'semantic': r['semantic_score']
            }
            for i, r in enumerate(ranked)
        ]
    }

    rankings_path = output_dir / 'quality_rankings.json'
    with open(rankings_path, 'w') as f:
        json.dump(rankings, f, indent=2)

    return summary, ranked


def run_evaluation():
    print("=" * 50)
    print("  3D MODEL EVALUATION PIPELINE")
    print("=" * 50)

    try:
        import numpy
        import PIL
    except ImportError:
        logger.error("Critical dependencies missing (numpy/PIL).")
        return

    # Load heavy models
    load_ai_models()

    output_dir = Path('output_models')
    input_dir = Path('input_images')

    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        return

    logger.info(f"\nScanning workspace...")
    models = model_discovery.discover_model_folders(output_dir, input_dir)

    if not models:
        logger.warning("No models found to evaluate.")
        return

    logger.info(f"Found {len(models)} models.")

    results = []
    eval_dir = Path('evaluation')
    eval_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nStarting evaluation...")
    for i, m in enumerate(models, 1):
        print(f"\n[{i}/{len(models)}] {m['name']}")
        res = evaluate_single_model(m)
        results.append(res)

    summary, ranked = save_reports(results, eval_dir)

    print("\n" + "=" * 50)
    print("  RESULTS SUMMARY")
    print("=" * 50)
    print(f"Total evaluated: {summary['total']}")
    print(f"Success rate:    {summary['successful']}/{summary['total']}")
    print(f"Mean Quality:    {summary['statistics']['mean_quality']:.1f}/100")
    print(f"Best Score:      {summary['statistics']['max_quality']:.1f}")

    if ranked:
        print("\nTop 3 Models:")
        for i, r in enumerate(ranked[:3], 1):
            print(f"  {i}. {r['name']:<20} | Score: {r['quality_score']:>4.1f}")

    print(f"\nFull reports saved to: {eval_dir}/")


if __name__ == "__main__":
    run_evaluation()
