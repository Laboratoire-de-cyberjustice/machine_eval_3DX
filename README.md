# 3D Model Evaluation Pipeline

This project evaluates 3D model generation quality by comparing renders to original images using various metrics.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd machine_eval_3DX
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```


## Project Structure

```
machine_eval_3DX/
├── generating_3D_object/       # 3D Model Generation Pipeline
│   ├── __init__.py
│   ├── config.yaml             # Generation configuration
│   └── pipeline.py             # Main generation script
├── machine_eval_process/       # Evaluation Logic
│   ├── __init__.py
│   ├── main_eval.py            # Main evaluation script
│   ├── metrics_image.py        # Image quality metrics (LPIPS, CLIP, etc.)
│   ├── metrics_mesh.py         # 3D mesh metrics
│   ├── model_discovery.py      # File discovery utilities
│   └── score_calculator.py     # Aggregation logic
├── input_images/               # Input 2D images
├── output_models/              # Generated 3D models & previews
├── evaluation/                 # Evaluation reports
├── .env                        # Environment variables (API Key)
├── .gitignore
├── requirements.txt
└── README.md
```

## Usage

1.  **Generate 3D Models:**
    -   Configure `generating_3D_object/config.yaml` if needed.
    -   Ensure your `.env` file has the `TRIPO_API_KEY`.
    -   Run the generation pipeline:
        ```bash
        python -m generating_3D_object.pipeline
        ```

2.  **Run the evaluation:**
    -   Place original images in `input_images/`.
    -   Place generated 3D models (GLB) and previews (PNG) in `output_models/`.
    ```bash
    python -m machine_eval_process.main_eval
    ```

3.  **Check results:**
    -   Detailed reports are saved in `evaluation/`.

## Metrics
-   **Quality:** Combined score based on shape, appearance, and semantic alignment.
-   **LPIPS:** Perceptual similarity.
-   **CLIP:** Semantic similarity.
-   **SSIM/PSNR:** Structural similarity and noise ratio.
-   **IoU:** Silhouette Intersection over Union.
