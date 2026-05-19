"""
gradcam_compare.py  --  Infinity Shield O1 (XAI component)

Compares EigenCAM vs AblationCAM heatmaps for YOLOv8n on a few test
images, to see whether AblationCAM localises on the ball better.

This is a TIME-BOXED experiment:
    - EigenCAM   : class-agnostic, fast, highlights dominant features.
    - AblationCAM: class-targeted, slow, *should* localise on the
                   detected object -- but is fiddly on detectors.

If AblationCAM is not clearly better, we keep EigenCAM and move on.

Setup:
    pip install grad-cam

Run from project root, virtualenv active:
    python notebooks/gradcam_compare.py
"""

from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from pytorch_grad_cam import EigenCAM, AblationCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


# ============================================================================
# CONFIG
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS      = PROJECT_ROOT / "models" / "checkpoints" / "yolov8n_plastic_v1.pt"
TEST_IMAGES  = PROJECT_ROOT / "data" / "annotated" / "test" / "images"
OUTPUT_DIR   = PROJECT_ROOT / "results" / "o1_detection" / "gradcam_compare"

NUM_IMAGES  = 3       # small first -- AblationCAM is slow
IMG_SIZE    = 640
CONF_THRESH = 0.25

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# YOLO OUTPUT WRAPPER
# ============================================================================

class YoloCamWrapper(torch.nn.Module):
    """Flattens YOLO's tuple output to a plain tensor for grad-cam."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        output = self.model(x)
        if isinstance(output, (tuple, list)):
            output = output[0]
        return output.reshape(output.shape[0], -1)


# ============================================================================
# ABLATIONCAM TARGET
# ============================================================================

class MeanScoreTarget:
    """
    AblationCAM needs a 'target' -- a number it watches while it
    switches off channels. A bigger drop in this number = that
    channel mattered more.

    For a detector there is no single class score, so we use the
    mean of the flattened output as a stand-in for 'overall
    detection signal'. It is a pragmatic target: not perfectly
    ball-specific, but it does make AblationCAM respond to where
    the detector's output is strongest.
    """
    def __call__(self, model_output):
        return model_output.mean()


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n=== Infinity Shield O1 -- EigenCAM vs AblationCAM (YOLOv8n) ===\n")

    yolo = YOLO(str(WEIGHTS))
    model = yolo.model
    model.eval()

    wrapped = YoloCamWrapper(model)
    wrapped.eval()

    target_layers = [model.model[-2]]
    print(f"[INFO] CAM target layer: model.model[-2]")
    print(f"[INFO] Testing on {NUM_IMAGES} images (AblationCAM is slow)\n")

    eigen_cam    = EigenCAM(model=wrapped, target_layers=target_layers)
    ablation_cam = AblationCAM(model=wrapped, target_layers=target_layers)

    image_paths = sorted(
        [p for p in TEST_IMAGES.iterdir()
         if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    )[:NUM_IMAGES]

    if not image_paths:
        print(f"[ERROR] No images found in {TEST_IMAGES}")
        return

    targets = [MeanScoreTarget()]

    for i, img_path in enumerate(image_paths, 1):
        print(f"  [{i}/{len(image_paths)}] {img_path.name} ...")

        bgr = cv2.imread(str(img_path))
        bgr = cv2.resize(bgr, (IMG_SIZE, IMG_SIZE))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb_norm = rgb.astype(np.float32) / 255.0
        input_tensor = torch.from_numpy(rgb_norm).permute(2, 0, 1).unsqueeze(0)

        # --- EigenCAM (fast, class-agnostic) ---
        print(f"        running EigenCAM ...")
        eigen_map = eigen_cam(input_tensor=input_tensor)[0]
        eigen_overlay = show_cam_on_image(rgb_norm, eigen_map, use_rgb=True)

        # --- AblationCAM (slow, class-targeted) ---
        print(f"        running AblationCAM (slow) ...")
        ablation_map = ablation_cam(input_tensor=input_tensor, targets=targets)[0]
        ablation_overlay = show_cam_on_image(rgb_norm, ablation_map, use_rgb=True)

        # --- Side-by-side: original | EigenCAM | AblationCAM ---
        original_bgr = cv2.cvtColor((rgb_norm * 255).astype(np.uint8),
                                    cv2.COLOR_RGB2BGR)
        eigen_bgr    = cv2.cvtColor(eigen_overlay,    cv2.COLOR_RGB2BGR)
        ablation_bgr = cv2.cvtColor(ablation_overlay, cv2.COLOR_RGB2BGR)

        # Label each panel
        for img, label in [(original_bgr, "ORIGINAL"),
                            (eigen_bgr, "EIGENCAM"),
                            (ablation_bgr, "ABLATIONCAM")]:
            cv2.putText(img, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 255, 255), 2)

        panel = np.hstack([original_bgr, eigen_bgr, ablation_bgr])
        out_path = OUTPUT_DIR / f"compare_{i:02d}_{img_path.stem}.jpg"
        cv2.imwrite(str(out_path), panel)
        print(f"        saved -> {out_path.name}\n")

    print(f"[Done] Comparison panels saved in: {OUTPUT_DIR}")
    print("\nHow to judge:")
    print("  Compare the EIGENCAM and ABLATIONCAM panels.")
    print("  If AblationCAM's warm region sits noticeably closer to the")
    print("  ball than EigenCAM's, it is worth using. If they look")
    print("  about the same (or AblationCAM is just noisier), keep")
    print("  EigenCAM -- it is much faster and simpler.")


if __name__ == "__main__":
    main()