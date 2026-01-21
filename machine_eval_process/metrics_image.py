import logging
import numpy as np
import traceback
from typing import Dict, Optional, Tuple, Any
from PIL import Image

logger = logging.getLogger(__name__)


def calculate_basic_metrics(
    original_path: str, 
    preview_path: str,
    lpips_model: Any = None,
    clip_model: Any = None,
    clip_preprocess: Any = None
) -> Dict:
    """
    Compare original photo and render.
    """
    metrics: Dict[str, object] = {}

    # ---- Load & align images ----
    try:
        orig = Image.open(original_path).convert("RGB")
        render = Image.open(preview_path).convert("RGB")
    except Exception as e:
        logger.error(f"Image load error: {e}")
        return metrics

    # Make them the same size (use render size as reference)
    try:
        render_size = render.size
        # Resize original to match render
        orig = orig.resize(render_size, Image.Resampling.BICUBIC)

        orig_np = np.asarray(orig)
        render_np = np.asarray(render)
    except Exception as e:
        logger.error(f"Image resize error: {e}")
        return metrics

    # ---- Helper: simple foreground bbox on the render ----
    def _foreground_bbox(img_np, bg_threshold: int = 15, min_area: int = 128) -> Optional[Tuple[int, int, int, int]]:
        gray = img_np.mean(axis=2)
        mask = gray > bg_threshold
        ys, xs = np.where(mask)
        if ys.size == 0:
            return None

        y1, y2 = ys.min(), ys.max()
        x1, x2 = xs.min(), xs.max()

        if (x2 - x1) * (y2 - y1) < min_area:
            return None

        # Add margin
        margin = 10
        h, w = img_np.shape[:2]
        x1 = max(0, int(x1 - margin))
        y1 = max(0, int(y1 - margin))
        x2 = min(w, int(x2 + margin))
        y2 = min(h, int(y2 + margin))
        return x1, y1, x2, y2

    # ---- Full-image metrics ----
    diff_full = orig_np.astype("float32") - render_np.astype("float32")
    metrics["mse_full"] = float(np.mean(diff_full ** 2))
    metrics["mean_pixel_diff_full"] = float(np.mean(np.abs(diff_full)))

    # Histogram Correlation
    hist_corrs = []
    for ch in range(3):
        h1 = np.histogram(orig_np[:, :, ch], bins=32, range=(0, 256))[0]
        h2 = np.histogram(render_np[:, :, ch], bins=32, range=(0, 256))[0]
        # Basic check to avoid divide-by-zero
        if h1.sum() > 0 and h2.sum() > 0:
            hist_corrs.append(np.corrcoef(h1, h2)[0, 1])
        else:
            hist_corrs.append(0.0)

    metrics["color_similarity_full"] = float(np.mean(hist_corrs)) if hist_corrs else 0.0

    # SSIM / PSNR
    try:
        from skimage.metrics import structural_similarity as ssim
        from skimage.metrics import peak_signal_noise_ratio as psnr

        metrics["ssim_full"] = float(ssim(orig_np, render_np, channel_axis=2, data_range=255))
        metrics["psnr_full"] = float(psnr(orig_np, render_np, data_range=255))
    except (ImportError, Exception):
        pass

    # ---- ROI metrics (Strategy 2) ----
    bbox = _foreground_bbox(render_np)
    metrics["roi_bbox"] = list(bbox) if bbox else None

    if bbox:
        x1, y1, x2, y2 = bbox
        orig_roi = orig_np[y1:y2, x1:x2]
        render_roi = render_np[y1:y2, x1:x2]

        diff_roi = orig_roi.astype("float32") - render_roi.astype("float32")
        metrics["mse_roi"] = float(np.mean(diff_roi ** 2))
        metrics["mean_pixel_diff_roi"] = float(np.mean(np.abs(diff_roi)))

        # Color
        hist_corrs_roi = []
        for ch in range(3):
            h1 = np.histogram(orig_roi[:, :, ch], bins=32, range=(0, 256))[0]
            h2 = np.histogram(render_roi[:, :, ch], bins=32, range=(0, 256))[0]
            if h1.sum() > 0 and h2.sum() > 0:
                hist_corrs_roi.append(np.corrcoef(h1, h2)[0, 1])
            else:
                hist_corrs_roi.append(0.0)
        metrics["color_similarity_roi"] = float(np.mean(hist_corrs_roi)) if hist_corrs_roi else 0.0

        try:
            from skimage.metrics import structural_similarity as ssim
            from skimage.metrics import peak_signal_noise_ratio as psnr
            metrics["ssim_roi"] = float(ssim(orig_roi, render_roi, channel_axis=2, data_range=255))
            metrics["psnr_roi"] = float(psnr(orig_roi, render_roi, data_range=255))
        except (ImportError, Exception):
            pass

    # ---- Silhouette IoU ----
    try:
        # Simple thresholding
        gray_render = render_np.mean(axis=2)
        gray_orig = orig_np.mean(axis=2)

        # Render is object-on-dark
        mask_render = gray_render > 15

        # Original is often lighter object
        # Heuristic: pixels brighter than mean + std
        mask_orig = gray_orig > (gray_orig.mean() + 0.3 * gray_orig.std())

        # Canonical view alignment (very loose approximation)
        def _get_canonical_resized(mask):
            ys, xs = np.where(mask)
            if ys.size == 0: return None

            y1, y2 = ys.min(), ys.max()
            x1, x2 = xs.min(), xs.max()

            # Crop
            sub = mask[y1:y2 + 1, x1:x2 + 1]

            # Resize mask to standard 128x128 to compare shapes roughly
            pil_mask = Image.fromarray((sub.astype('uint8') * 255))
            pil_mask = pil_mask.resize((128, 128), Image.Resampling.NEAREST)
            return np.array(pil_mask) > 127

        can_render = _get_canonical_resized(mask_render)
        can_orig = _get_canonical_resized(mask_orig)

        if can_render is not None and can_orig is not None:
            inter = (can_render & can_orig).sum()
            union = (can_render | can_orig).sum()
            if union > 0:
                metrics["silhouette_iou"] = float(inter / union)

    except Exception as e:
        logger.warning(f"IoU error: {e}")

    # ---- LPIPS (Perceptual) ----
    if lpips_model is not None:
        try:
            import torch
            def _to_tensor(x):
                t = torch.from_numpy(x).permute(2, 0, 1).float()
                t = t / 127.5 - 1.0  # [-1, 1]
                return t.unsqueeze(0)

            with torch.no_grad():
                d = lpips_model(_to_tensor(orig_np), _to_tensor(render_np))
                metrics["lpips_full"] = float(d.item())
        except Exception as e:
            logger.warning(f"LPIPS error: {e}")

    # ---- Semantic (CLIP) ----
    if clip_model is not None and clip_preprocess is not None:
        try:
            import torch

            # Preprocess needs PIL
            img1 = clip_preprocess(orig).unsqueeze(0)
            img2 = clip_preprocess(render).unsqueeze(0)

            with torch.no_grad():
                feat1 = clip_model.encode_image(img1)
                feat2 = clip_model.encode_image(img2)

                feat1 /= feat1.norm(dim=-1, keepdim=True)
                feat2 /= feat2.norm(dim=-1, keepdim=True)

                sim = (feat1 @ feat2.T).item()
                metrics["semantic_similarity"] = float(sim)
        except Exception as e:
            logger.warning(f"CLIP error: {e}")

    return metrics
