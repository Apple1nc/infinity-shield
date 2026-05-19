"""
gradcam_mobilenet.py  --  Infinity Shield O1 (XAI component)

Generates EigenCAM saliency maps for the trained MobileNet-SSD
detector, with the same three-panel layout used for YOLOv8n:
    original | detection boxes | EigenCAM heatmap

This lets the two detectors' explainability be compared directly.

Why EigenCAM (same reasoning as the YOLO script):
    EigenCAM needs no class-specific target, so it works on object
    detectors without the fragile targeting that broke AblationCAM.
    It highlights the layer's dominant feature regions.

Note:
    This script imports the model + dataset definitions from
    benchmark_mobilenet.py, so that file must be the Option A
    version (full fine-tuning, no backbone freeze) -- the same one
    the saved weights came from.

Setup:
    pip install grad-cam

Run from project root, virtualenv active:
    python notebooks/gradcam_mobilenet.py
"""

from pathlib import Path

import cv2
import numpy as np
import torch
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# Reuse the exact model + config from the benchmark script
from benchmark_mobilenet import build_model, DEVICE, NUM_CLASSES


# ============================================================================
# CONFIG
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS      = PROJECT_ROOT / "models" / "checkpoints" / "mobilenet_ssd_plastic_v1.pt"
TEST_IMAGES  = PROJECT_ROOT / "data" / "annotated" / "test" / "images"
OUTPUT_DIR   = PROJECT_ROOT / "results" / "o1_detection" / "gradcam_mobilenet"

NUM_IMAGES  = 8
IMG_SIZE    = 320     # SSDLite native input size
CONF_THRESH = 0.25    # detection confidence threshold

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# SSD OUTPUT WRAPPER
# ============================================================================

class SsdCamWrapper(torch.nn.Module):
    """
    Why this exists:
        In eval mode, torchvision's SSD returns a list of dicts
        ({'boxes':..., 'scores':..., 'labels':...}) -- grad-cam cannot
        digest that. This wrapper runs only the FEATURE part of the
        model and returns a flat tensor, which is all EigenCAM needs
        (EigenCAM works from layer activations, not final scores).
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # Run just the backbone feature extractor
        features = self.model.backbone(x)
        # features is an ordered dict of feature maps; take the first,
        # flatten it to [batch, N] so grad-cam can index it
        first = list(features.values())[0]
        return first.reshape(first.shape[0], -1)


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n=== Infinity Shield O1 -- EigenCAM for MobileNet-SSD ===\n")

    # --- Build model + load trained weights ---
    model = build_model()
    model.load_state_dict(torch.load(WEIGHTS, map_location=DEVICE))
    model.eval()
    print(f"[INFO] Loaded weights: {WEIGHTS.name}")

    # --- Target layer: last block of the MobileNet backbone ---
    # The backbone's features carry the richest object cues. We target
    # the final feature block before detection.
    target_layer = model.backbone.features[-1]
    print(f"[INFO] CAM target layer: backbone.features[-1]")
    print(f"[INFO] Detection confidence threshold: {CONF_THRESH}\n")

    wrapped = SsdCamWrapper(model)
    wrapped.eval()
    cam = EigenCAM(model=wrapped, target_layers=[target_layer])

    # --- Collect test images ---
    image_paths = sorted(
        [p for p in TEST_IMAGES.iterdir()
         if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    )[:NUM_IMAGES]

    if not image_paths:
        print(f"[ERROR] No images found in {TEST_IMAGES}")
        return

    total_detections = 0
    images_with_detection = 0

    for i, img_path in enumerate(image_paths, 1):
        # --- Load + resize ---
        bgr = cv2.imread(str(img_path))
        bgr = cv2.resize(bgr, (IMG_SIZE, IMG_SIZE))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb_norm = rgb.astype(np.float32) / 255.0
        input_tensor = torch.from_numpy(rgb_norm).permute(2, 0, 1).unsqueeze(0)

        # --- STEP 1: run detection and report what it found ---
        with torch.no_grad():
            predictions = model(input_tensor.to(DEVICE))[0]

        scores = predictions["scores"].cpu().numpy()
        boxes  = predictions["boxes"].cpu().numpy()
        keep   = scores >= CONF_THRESH
        boxes, scores = boxes[keep], scores[keep]

        n_det = len(boxes)
        total_detections += n_det
        if n_det > 0:
            images_with_detection += 1
            conf_str = [f"{s:.2f}" for s in scores]
            print(f"  [{i}/{len(image_paths)}] {img_path.name}")
            print(f"        detected {n_det} ball(s), confidence: {conf_str}")
        else:
            print(f"  [{i}/{len(image_paths)}] {img_path.name}")
            print(f"        detected NO balls")

        # --- Draw detection boxes (middle panel) ---
        detection_img = (rgb_norm * 255).astype(np.uint8).copy()
        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(detection_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(detection_img, f"ball {score:.2f}",
                        (x1, max(y1 - 5, 12)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (0, 255, 0), 1)

        # --- STEP 2: EigenCAM heatmap ---
        grayscale_cam = cam(input_tensor=input_tensor)[0]
        overlay = show_cam_on_image(rgb_norm, grayscale_cam, use_rgb=True)

        # --- Save three-panel: original | detection | heatmap ---
        original_bgr  = cv2.cvtColor((rgb_norm * 255).astype(np.uint8),
                                     cv2.COLOR_RGB2BGR)
        detection_bgr = cv2.cvtColor(detection_img, cv2.COLOR_RGB2BGR)
        overlay_bgr   = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        panel = np.hstack([original_bgr, detection_bgr, overlay_bgr])

        out_path = OUTPUT_DIR / f"gradcam_{i:02d}_{img_path.stem}.jpg"
        cv2.imwrite(str(out_path), panel)

    # --- Summary ---
    print(f"\n[Done] Panels saved in: {OUTPUT_DIR}")
    print("\n=== DETECTION SUMMARY (MobileNet-SSD) ===")
    print(f"  Images processed     : {len(image_paths)}")
    print(f"  Images with a ball   : {images_with_detection} / {len(image_paths)}")
    print(f"  Total balls detected : {total_detections}")
    print("\nPanels (left -> right): original | detection | EigenCAM heatmap")
    print("Compare this against the YOLOv8n panels -- the detection")
    print("counts and heatmap focus side by side are the O1 XAI comparison.")


if __name__ == "__main__":
    main()