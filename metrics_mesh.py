import logging
import traceback
from typing import Dict, Any

logger = logging.getLogger(__name__)

def load_glb_as_mesh(model_path: str):
    try:
        import trimesh
    except ImportError:
        logger.error("trimesh not installed.")
        return None

    loaded = trimesh.load(model_path)

    if isinstance(loaded, trimesh.Scene):
        if len(loaded.geometry) > 0:
            mesh = list(loaded.geometry.values())[0]
            return mesh
        else:
            raise ValueError("Scene has no geometry")
    return loaded


def analyze_model(model_path: str) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}

    try:
        import trimesh
    except ImportError:
        return metrics

    try:
        mesh = load_glb_as_mesh(model_path)
        if mesh is None:
            return metrics

        metrics['vertices'] = int(len(mesh.vertices))
        metrics['faces'] = int(len(mesh.faces))
        metrics['watertight'] = bool(mesh.is_watertight)

        has_tex = False
        if hasattr(mesh.visual, 'material') and mesh.visual.material is not None:
             has_tex = True
        metrics['has_texture'] = has_tex

        bounds = mesh.bounds
        size = bounds[1] - bounds[0]
        metrics['size_x'] = float(size[0])
        metrics['size_y'] = float(size[1])
        metrics['size_z'] = float(size[2])

        metrics['surface_area'] = float(mesh.area)

        mesh_flags = []

        try:
            if mesh.is_self_intersecting:
                metrics['self_intersecting'] = True
                mesh_flags.append('self_intersecting')
            else:
                metrics['self_intersecting'] = False
        except Exception:
            metrics['self_intersecting'] = None

        try:
            non_manifold = getattr(mesh, "edges_non_manifold", None)
            if non_manifold is not None:
                count = int(non_manifold.shape[0])
                metrics['non_manifold_edges'] = count
                if count > 0:
                    mesh_flags.append('non_manifold_edges')
        except Exception:
            metrics['non_manifold_edges'] = None

        if not metrics['watertight']:
            mesh_flags.append('not_watertight')

        metrics['mesh_flags'] = mesh_flags

    except Exception as e:
        logger.error(f"Model analysis failed: {e}")

    return metrics

