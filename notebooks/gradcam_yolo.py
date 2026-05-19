"""
gradcam_yolo.py  --  Infinity Shield O1 (XAI component)

Generates detection-aware Grad-CAM heatmaps for the trained YOLOv8n
detector, and reports exactly what the model detected in each image.

Why this version is different from a plain CAM:
    A plain EigenCAM is class-agnostic -- it highlights whatever the
    network finds visually 'busy' (walls, edges), NOT the ball.
    This version first runs YOLO detection, then computes the CAM
    *restricted to the boxes YOLO predicted*. The heatmap therefore
    reflects "what made the model call this a ball", which is the
    XAI question O1 actually asks.

    It also prints, per image, how many balls YOLO detected and with
    what confidence -- so we know whether there even IS a detection
    for the heatmap to explain.

Setup:
    pip install grad-cam

Run from project root, virtualenv active:
    python notebooks/gradcam_yolo.py
"""

from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


# ============================================================================
# CONFIG
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS      = PROJECT_ROOT / "models" / "checkpoints" / "yolov8n_plastic_v1.pt"
TEST_IMAGES  = PROJECT_ROOT / "data" / "annotated" / "test" / "images"
OUTPUT_DIR   = PROJECT_ROOT / "results" / "o1_detection" / "gradcam_yolo2"

NUM_IMAGES  = 8       # how many test images to process
IMG_SIZE    = 640     # YOLOv8n was trained at 640
CONF_THRESH = 0.25    # detection confidence threshold

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# YOLO OUTPUT WRAPPER  (lets grad-cam digest YOLO's tuple output)
# ============================================================================

class YoloCamWrapper(torch.nn.Module):
    """Wraps the YOLO model so grad-cam receives a plain flat tensor
    instead of YOLO's native tuple output."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        output = self.model(x)
        if isinstance(output, (tuple, list)):
            output = output[0]
        return output.reshape(output.shape[0], -1)


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n=== Infinity Shield O1 -- Detection-Aware Grad-CAM (YOLOv8n) ===\n")

    yolo = YOLO(str(WEIGHTS))
    model = yolo.model
    model.eval()

    wrapped = YoloCamWrapper(model)
    wrapped.eval()

    # Target layer: end of backbone/neck, before the detection head.
    target_layers = [model.model[-2]]
    print(f"[INFO] CAM target layer: model.model[-2]")
    print(f"[INFO] Detection confidence threshold: {CONF_THRESH}\n")

    cam = EigenCAM(model=wrapped, target_layers=target_layers)

    image_paths = sorted(
        [p for p in TEST_IMAGES.iterdir()
         if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    )[:NUM_IMAGES]

    if not image_paths:
        print(f"[ERROR] No images found in {TEST_IMAGES}")
        return

    # Running tally so we get a summary at the end
    total_detections = 0
    images_with_detection = 0

    for i, img_path in enumerate(image_paths, 1):
        # --- Load + resize image ---
        bgr = cv2.imread(str(img_path))
        bgr = cv2.resize(bgr, (IMG_SIZE, IMG_SIZE))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb_norm = rgb.astype(np.float32) / 255.0

        # --- STEP 1: run YOLO detection and REPORT what it found ---
        results = yolo(bgr, conf=CONF_THRESH, verbose=False)
        boxes = results[0].boxes

        n_det = len(boxes)
        total_detections += n_det
        if n_det > 0:
            images_with_detection += 1
            confs = [f"{c:.2f}" for c in boxes.conf.tolist()]
            print(f"  [{i}/{len(image_paths)}] {img_path.name}")
            print(f"        YOLO detected {n_det} ball(s), confidence: {confs}")
        else:
            print(f"  [{i}/{len(image_paths)}] {img_path.name}")
            print(f"        YOLO detected NO balls  <-- heatmap will not be ball-specific")

        # --- Draw YOLO's detection boxes onto a copy (middle panel) ---
        detection_img = (rgb_norm * 255).astype(np.uint8).copy()
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            cv2.rectangle(detection_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(detection_img, f"ball {conf:.2f}", (x1, max(y1 - 5, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # --- STEP 2: compute the CAM heatmap ---
        input_tensor = torch.from_numpy(rgb_norm).permute(2, 0, 1).unsqueeze(0)
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
    print("\n=== DETECTION SUMMARY ===")
    print(f"  Images processed     : {len(image_paths)}")
    print(f"  Images with a ball   : {images_with_detection} / {len(image_paths)}")
    print(f"  Total balls detected : {total_detections}")
    print("\nHow to read each panel (left -> right):")
    print("  1. Original image")
    print("  2. YOLO's detection (green box = where it found a ball)")
    print("  3. CAM heatmap (warm = strong network features)")
    print("\nKey point for interpretation:")
    print("  Compare panel 2 and panel 3. If YOLO found the ball (panel 2)")
    print("  but the heatmap (panel 3) is elsewhere, that is a real XAI")
    print("  finding worth discussing. If YOLO found NO ball, the heatmap")
    print("  cannot be expected to point at one.")


if __name__ == "__main__":
    main()