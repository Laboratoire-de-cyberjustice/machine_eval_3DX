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

## Usage

1.  **Prepare your data:**
    -   Place original images in `input_images/`.
    -   Place generated 3D models (GLB) and previews (PNG) in `output_models/`.

2.  **Run the evaluation:**
    ```bash
    python main_eval.py
    ```

3.  **Check results:**
    -   Detailed reports are saved in `evaluation/`.

## Metrics
-   **Quality:** Combined score based on shape, appearance, and semantic alignment.
-   **LPIPS:** Perceptual similarity.
-   **CLIP:** Semantic similarity.
-   **SSIM/PSNR:** Structural similarity and noise ratio.
-   **IoU:** Silhouette Intersection over Union.
