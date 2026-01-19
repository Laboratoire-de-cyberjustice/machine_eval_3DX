from typing import Dict


def calculate_scores(metrics: Dict) -> Dict[str, float]:
    shape_score = None
    appearance_score = None
    semantic_score = None

    # --- Shape score: Silhouette IoU in [0,1] -> [0,100] ---
    iou = metrics.get("silhouette_iou")
    if iou is not None:
        shape_score = round(max(0.0, min(1.0, float(iou))) * 100.0, 2)

    # --- Appearance score: ROI SSIM + color + LPIPS ---
    # Prefer ROI metrics, fall back to full
    has_roi = ("ssim_roi" in metrics) or ("color_similarity_roi" in metrics)
    prefix = "roi" if has_roi else "full"

    def get_metric(name, default=None):
        key = f"{name}_{prefix}"
        return metrics.get(key, default)

    acc = 0.0
    w = 0.0

    ssim_v = get_metric("ssim")
    if ssim_v is not None:
        acc += float(ssim_v) * 100.0  # [0,1] -> [0,100]
        w += 1.5

    color_v = get_metric("color_similarity")
    if color_v is not None:
        # [-1,1] -> [0,100]
        acc += ((float(color_v) + 1.0) / 2.0) * 100.0
        w += 1.0

    psnr_v = get_metric("psnr")
    if psnr_v is not None:
        # map ~10–30 dB to 0–100
        psnr_norm = max(0.0, min(1.0, (float(psnr_v) - 10.0) / 20.0))
        acc += psnr_norm * 100.0
        w += 0.5

    lpips_v = metrics.get("lpips_full")
    if lpips_v is not None:
        # clamp LPIPS [0,0.5]; invert to score
        lp = float(lpips_v)
        lp_norm = max(0.0, min(0.5, lp)) / 0.5
        lp_score = (1.0 - lp_norm) * 100.0
        acc += lp_score
        w += 1.5

    if w > 0.0:
        appearance_score = round(acc / w, 2)

    # --- Semantic score: CLIP similarity in [-1,1] -> [0,100] ---
    sem = metrics.get("semantic_similarity")
    if sem is not None:
        sem = float(sem)
        sem_norm = (sem + 1.0) / 2.0  # -1..1 -> 0..1
        semantic_score = round(max(0.0, min(1.0, sem_norm)) * 100.0, 2)

    # --- Combine into final quality score ---
    # Default neutral scores if missing
    s = shape_score if shape_score is not None else 50.0
    a = appearance_score if appearance_score is not None else 50.0
    se = semantic_score if semantic_score is not None else 50.0

    # Base combination (normalize available weights)
    weights = []
    values = []
    if shape_score is not None:
        weights.append(0.4)
        values.append(s)
    if appearance_score is not None:
        weights.append(0.4)
        values.append(a)
    if semantic_score is not None:
        weights.append(0.2)
        values.append(se)

    if not weights:
        quality = 50.0
    else:
        total_w = sum(weights)
        quality = sum(v * (w / total_w) for v, w in zip(values, weights))

    # --- Mesh-health penalties ---
    mesh_flags = metrics.get("mesh_flags") or []
    penalty = 0.0
    if 'not_watertight' in mesh_flags:
        penalty += 10.0
    if 'self_intersecting' in mesh_flags:
        penalty += 15.0
    if 'non_manifold_edges' in mesh_flags:
        penalty += 10.0

    quality = max(0.0, min(100.0, quality - penalty))

    return {
        'shape_score': round(shape_score if shape_score is not None else 50.0, 2),
        'appearance_score': round(appearance_score if appearance_score is not None else 50.0, 2),
        'semantic_score': round(semantic_score if semantic_score is not None else 50.0, 2),
        'quality_score': round(quality, 2),
    }
