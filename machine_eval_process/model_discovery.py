import os
from pathlib import Path
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


def discover_model_folders(output_dir: Path, input_dir: Path, exclude_folders: List[str] = None) -> List[Dict]:
    if exclude_folders is None:
        exclude_folders = ['evaluation', 'testing_batch_1']

    models = []

    for root, dirs, files in os.walk(output_dir):
        root_path = Path(root)

        if any(ex in root_path.parts for ex in exclude_folders):
            continue

        glb_files = [f for f in files if f.endswith('.glb')]
        if not glb_files:
            continue

        folder = root_path
        glb_file = folder / glb_files[0]

        png_files = list(folder.glob('*.png'))
        if not png_files:
            continue
        preview_file = png_files[0]

        try:
            rel_path = folder.relative_to(output_dir)
        except ValueError:
            continue
    
        category = None
        if folder.parent != output_dir:
            category = folder.parent.name

        model_name = folder.name

        image_file = None

        candidates = []

        if category:
            for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                candidates.append(input_dir / category / f"{model_name}{ext}")
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            candidates.append(input_dir / f"{model_name}{ext}")

        for candidate in candidates:
            if candidate.exists():
                image_file = candidate
                break

        if not image_file:
            continue

        models.append({
            'folder': folder,
            'name': model_name,
            'category': category,
            'glb': glb_file,
            'image': image_file,
            'preview': preview_file
        })
    return models
