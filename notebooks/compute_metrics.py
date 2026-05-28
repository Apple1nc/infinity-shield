"""
compute_metrics.py  --  Infinity Shield O1 (unified metrics)

Computes the SAME COCO-style detection metrics for all three trained
detectors, using a single torchmetrics implementation. This replaces
the situation where YOLOv8n's metrics came from Ultralytics and the
torchvision models' came from a hand-written counting loop, which
were not directly comparable.

Outputs:
    Model        | mAP_50_95 | mAP_50 | mAR_100 | Latency_mean_ms | Latency_p95_ms
where:
    mAP_50_95 = COCO-style mean Average Precision, averaged over
                IoU thresholds 0.50:0.05:0.95  (the headline COCO number)
    mAP_50    = mean Average Precision at IoU 0.50              (closer to old "mAP")
    mAR_100   = mean Average Recall, top 100 predictions per image

All three models share this same evaluation code, so the rows are
directly comparable.

Setup:
    pip install torchmetrics pycocotools

Usage (one model at a time):
    python notebooks/compute_metrics.py yolo
    python notebooks/compute_metrics.py mobilenet
    python notebooks/compute_metrics.py fasterrcnn

Run all three, then open the CSV and you have the final O1 table.
"""

import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision


# ============================================================================
# CONFIG
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data" / "annotated"
WEIGHTS_DIR  = PROJECT_ROOT / "models" / "checkpoints"
RESULTS_CSV  = PROJECT_ROOT / "results" / "o1_detection" / "detection_results_unified.csv"

DEVICE = torch.device("cpu")

MODEL_FILES = {
    "yolo":       WEIGHTS_DIR / "yolov8n_plastic_v1.pt",
    "mobilenet":  WEIGHTS_DIR / "mobilenet_ssd_plastic_v1.pt",
    "fasterrcnn": WEIGHTS_DIR / "fasterrcnn_plastic_v1.pt",
}

MODEL_LABELS = {
    "yolo":       "YOLOv8n",
    "mobilenet":  "MobileNet-SSD",
    "fasterrcnn": "Faster R-CNN",
}

LATENCY_INPUT_SIZE = {
    "yolo":       640,
    "mobilenet":  320,
    "fasterrcnn": 640,
}

RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)


# ============================================================================
# DATASET  (same logic as the benchmark scripts)
# ============================================================================

def load_test_set():
    """
    Yields one (image_tensor, ground_truth_dict) per test image.
    Ground truth dict: {boxes: [N,4] xyxy pixels, labels: [N] long}
    """
    img_dir   = DATA_DIR / "test" / "images"
    label_dir = DATA_DIR / "test" / "labels"

    samples = []
    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue

        img_bgr = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]
        img_t = TF.to_tensor(img_rgb)

        label_path = label_dir / (img_path.stem + ".txt")
        boxes, labels = [], []
        if label_path.exists():
            for line in label_path.read_text().splitlines():
                parts = line.split()
                if len(parts) != 5:
                    continue
                _, cx, cy, bw, bh = map(float, parts)
                x1 = (cx - bw / 2) * w
                y1 = (cy - bh / 2) * h
                x2 = (cx + bw / 2) * w
                y2 = (cy + bh / 2) * h
                boxes.append([x1, y1, x2, y2])
                labels.append(1)

        target = {
            "boxes":  torch.tensor(boxes,  dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.int64),
        }
        samples.append((img_t, target, img_path))

    return samples


# ============================================================================
# MODEL LOADERS  --  one per architecture, each returns a callable
# that takes an image tensor and returns torchmetrics-format predictions
# ============================================================================

def load_yolo():
    from ultralytics import YOLO
    yolo = YOLO(str(MODEL_FILES["yolo"]))

    def predict(img_tensor):
        # YOLO wants a numpy HWC image, not a tensor
        img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        results = yolo(img_np, verbose=False)[0]
        boxes  = results.boxes.xyxy.cpu()
        scores = results.boxes.conf.cpu()
        labels = torch.ones(len(boxes), dtype=torch.int64)  # single class
        return {"boxes": boxes, "scores": scores, "labels": labels}

    return predict


def load_torchvision_model(kind):
    """Loads MobileNet-SSD or Faster R-CNN from their benchmark scripts."""
    if kind == "mobilenet":
        from benchmark_mobilenet import build_model
    else:  # fasterrcnn
        from benchmark_fasterrcnn import build_model

    model = build_model()
    model.load_state_dict(torch.load(MODEL_FILES[kind], map_location=DEVICE))
    model.eval()

    def predict(img_tensor):
        with torch.no_grad():
            pred = model([img_tensor.to(DEVICE)])[0]
        # torchmetrics expects predictions with label != 0 (0 = background)
        # Both models already use class 1 = ball, so passthrough is fine.
        return {
            "boxes":  pred["boxes"].cpu(),
            "scores": pred["scores"].cpu(),
            "labels": pred["labels"].cpu(),
        }

    return predict, model   # return the model too, for latency measurement


# ============================================================================
# LATENCY  --  same protocol as the benchmark scripts
# ============================================================================

def measure_latency_yolo(yolo_predict, runs=100):
    """YOLO latency: feed it a dummy image, time the calls."""
    from ultralytics import YOLO
    yolo = YOLO(str(MODEL_FILES["yolo"]))
    dummy = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

    for _ in range(10):
        yolo(dummy, verbose=False)

    times = []
    for _ in range(runs):
        t = time.perf_counter()
        yolo(dummy, verbose=False)
        times.append((time.perf_counter() - t) * 1000)

    return float(np.mean(times)), float(np.percentile(times, 95))


def measure_latency_torchvision(model, input_size, runs=100):
    """torchvision detection latency."""
    model.eval()
    dummy = [torch.rand(3, input_size, input_size).to(DEVICE)]

    with torch.no_grad():
        for _ in range(10):
            model(dummy)

        times = []
        for _ in range(runs):
            t = time.perf_counter()
            model(dummy)
            times.append((time.perf_counter() - t) * 1000)

    return float(np.mean(times)), float(np.percentile(times, 95))


# ============================================================================
# MAIN
# ============================================================================

def main():
    if len(sys.argv) < 2 or sys.argv[1] not in MODEL_FILES:
        print("Usage: python notebooks/compute_metrics.py [yolo|mobilenet|fasterrcnn]")
        sys.exit(1)

    kind  = sys.argv[1]
    label = MODEL_LABELS[kind]

    weights = MODEL_FILES[kind]
    if not weights.exists():
        print(f"[ERROR] Weights file not found: {weights}")
        sys.exit(1)

    print(f"\n=== Infinity Shield O1 -- Unified Metrics ({label}) ===\n")

    # --- Load test set ---
    print("Loading test set ...")
    test_samples = load_test_set()
    print(f"  {len(test_samples)} test images loaded\n")

    # --- Load model ---
    print(f"Loading {label} ...")
    if kind == "yolo":
        predict_fn = load_yolo()
        tv_model = None
    else:
        predict_fn, tv_model = load_torchvision_model(kind)
    print(f"  Loaded weights: {weights.name}\n")

    # --- Run predictions on test set, collect for torchmetrics ---
    print("Running predictions on test set ...")
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox",
                                  class_metrics=False)

    for i, (img_tensor, target, img_path) in enumerate(test_samples, 1):
        pred = predict_fn(img_tensor)
        metric.update([pred], [target])
        if i % 10 == 0 or i == len(test_samples):
            print(f"  {i}/{len(test_samples)} images processed")

    # --- Compute metrics ---
    print("\nComputing COCO-style metrics ...")
    results = metric.compute()

    mAP_50_95 = float(results["map"])
    mAP_50    = float(results["map_50"])
    mAR_100   = float(results["mar_100"])

    print(f"  mAP@[.50:.95] : {mAP_50_95:.4f}")
    print(f"  mAP@.50       : {mAP_50:.4f}")
    print(f"  mAR@100       : {mAR_100:.4f}")

    # --- Latency benchmark ---
    print("\nMeasuring latency ...")
    if kind == "yolo":
        lat_mean, lat_p95 = measure_latency_yolo(predict_fn)
    else:
        lat_mean, lat_p95 = measure_latency_torchvision(
            tv_model, LATENCY_INPUT_SIZE[kind]
        )
    print(f"  Mean : {lat_mean:.1f} ms")
    print(f"  P95  : {lat_p95:.1f} ms")

    # --- Append to unified CSV ---
    file_exists = RESULTS_CSV.exists()
    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Model", "mAP_50_95", "mAP_50", "mAR_100",
                             "Latency_mean_ms", "Latency_p95_ms"])
        writer.writerow([
            label,
            round(mAP_50_95, 4),
            round(mAP_50, 4),
            round(mAR_100, 4),
            round(lat_mean, 2),
            round(lat_p95, 2),
        ])

    print(f"\n[Done] Row appended -> {RESULTS_CSV}")
    print("\n=== SUMMARY ===")
    print(f"  Model        : {label}")
    print(f"  mAP@[.5:.95] : {mAP_50_95:.4f}")
    print(f"  mAP@.5       : {mAP_50:.4f}")
    print(f"  mAR@100      : {mAR_100:.4f}")
    print(f"  Latency mean : {lat_mean:.1f} ms")
    print(f"  Latency p95  : {lat_p95:.1f} ms")


if __name__ == "__main__":
    main()
