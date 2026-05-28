"""
benchmark_fasterrcnn.py  --  Infinity Shield, Objective O1 (third detector)

Purpose:
    Fine-tune Faster R-CNN with a MobileNetV3-Large FPN backbone on
    the plastic-ball dataset and measure the same three metrics used
    for YOLOv8n and MobileNet-SSD:
        - recall              (did we find the balls that were there?)
        - false positive rate (how often did we flag something wrong?)
        - latency             (how many milliseconds per image?)

Why this detector:
    Replaces NanoDet (dependency conflict with torch >= 2.0). Reuses
    the same torchvision pipeline as MobileNet-SSD but is a TWO-STAGE
    detector (region proposals + per-region classification) rather
    than one-stage. This gives O1 a genuine one-stage vs two-stage
    architectural contrast under identical training data.

Run from project root, virtualenv active:
    python notebooks/benchmark_fasterrcnn.py
"""

import time
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import (
    fasterrcnn_mobilenet_v3_large_fpn,
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms.functional as TF


# ============================================================================
# CONFIG  --  same knobs as benchmark_mobilenet.py for fair comparison
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data" / "annotated"
RESULTS_DIR  = PROJECT_ROOT / "results" / "o1_detection"
WEIGHTS_DIR  = PROJECT_ROOT / "models" / "checkpoints"

EPOCHS      = 50      # match the MobileNet-SSD run for a fair comparison
BATCH_SIZE  = 4
LR          = 0.001
CONF_THRESH = 0.4
IOU_THRESH  = 0.4

# class 0 = background, class 1 = plastic_ball
NUM_CLASSES = 2

DEVICE = torch.device("cpu")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
torch.manual_seed(42)


# ============================================================================
# DATASET  --  identical to benchmark_mobilenet.py
# ============================================================================

class PlasticBallDataset(Dataset):
    """
    Reads the Roboflow YOLOv8 export. Same code path used by the
    MobileNet-SSD benchmark, so all three detectors see the same data.
    """

    def __init__(self, split):
        img_dir   = DATA_DIR / split / "images"
        label_dir = DATA_DIR / split / "labels"

        if not img_dir.exists():
            raise FileNotFoundError(f"Could not find {img_dir}")

        self.samples = []
        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() in (".jpg", ".jpeg", ".png"):
                label_path = label_dir / (img_path.stem + ".txt")
                self.samples.append((img_path, label_path))

        print(f"  [{split}] {len(self.samples)} images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label_path = self.samples[idx]

        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width = img.shape[:2]
        img_tensor = TF.to_tensor(img)

        boxes, labels = [], []
        if label_path.exists():
            for line in label_path.read_text().splitlines():
                parts = line.split()
                if len(parts) != 5:
                    continue
                _, cx, cy, bw, bh = map(float, parts)
                x1 = (cx - bw / 2) * width
                y1 = (cy - bh / 2) * height
                x2 = (cx + bw / 2) * width
                y2 = (cy + bh / 2) * height
                boxes.append([x1, y1, x2, y2])
                labels.append(1)

        target = {
            "boxes":  torch.tensor(boxes,  dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.int64),
        }
        return img_tensor, target


def collate_fn(batch):
    return tuple(zip(*batch))


# ============================================================================
# MODEL
# ============================================================================

def build_model():
    """
    Load Faster R-CNN with a MobileNetV3-Large FPN backbone, pretrained
    on COCO, then REPLACE the box predictor head for our NUM_CLASSES.

    Faster R-CNN is a two-stage detector:
        Stage 1: a Region Proposal Network suggests "interesting" boxes
        Stage 2: a classifier decides what (if anything) is in each box
    We only need to retrain the final classifier; the rest transfers.
    """
    weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
    model   = fasterrcnn_mobilenet_v3_large_fpn(weights=weights)

    # Swap the final classifier head for our class count
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)

    return model.to(DEVICE)


# ============================================================================
# TRAINING
# ============================================================================

def train_one_epoch(model, loader, optimizer, epoch):
    model.train()
    running_loss = 0.0

    for images, targets in loader:
        images  = [img.to(DEVICE) for img in images]
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        losses = sum(model(images, targets).values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        running_loss += losses.item()

    avg_loss = running_loss / len(loader)
    print(f"  Epoch {epoch:2d}/{EPOCHS}  -  avg loss {avg_loss:.4f}")
    return avg_loss


# ============================================================================
# EVALUATION  --  identical logic to benchmark_mobilenet.py
# ============================================================================

def iou(box_a, box_b):
    x1 = max(box_a[0], box_b[0]);  y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2]);  y2 = min(box_a[3], box_b[3])
    overlap = max(0, x2 - x1) * max(0, y2 - y1)
    area_a  = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b  = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union   = area_a + area_b - overlap
    return overlap / union if union > 0 else 0.0


def evaluate(model, loader, split_name):
    model.eval()
    tp = fp = fn = 0

    with torch.no_grad():
        for images, targets in loader:
            images      = [img.to(DEVICE) for img in images]
            predictions = model(images)

            for pred, target in zip(predictions, targets):
                real_boxes = target["boxes"].numpy()

                scores     = pred["scores"].cpu().numpy()
                pred_boxes = pred["boxes"].cpu().numpy()[scores >= CONF_THRESH]

                matched = set()
                for pb in pred_boxes:
                    best_iou, best_idx = 0.0, -1
                    for i, rb in enumerate(real_boxes):
                        if i in matched:
                            continue
                        score = iou(pb, rb)
                        if score > best_iou:
                            best_iou, best_idx = score, i

                    if best_iou >= IOU_THRESH:
                        tp += 1
                        matched.add(best_idx)
                    else:
                        fp += 1

                fn += len(real_boxes) - len(matched)

    recall  = tp / (tp + fn) if (tp + fn) else 0.0
    fp_rate = fp / (tp + fp) if (tp + fp) else 0.0
    print(f"  [{split_name}] TP={tp} FP={fp} FN={fn}  ->  "
          f"recall={recall:.3f}  FP-rate={fp_rate:.3f}")
    return recall, fp_rate


# ============================================================================
# LATENCY
# ============================================================================

def measure_latency(model, runs=100):
    """
    Same protocol used for YOLOv8n and MobileNet-SSD: 10 warmup,
    then 100 timed passes on a dummy image.
    Note: Faster R-CNN doesn't have a fixed input size; we use 640x640
    (matching YOLOv8n) so the latency comparison is fair-ish, though
    you should mention the resolution differences in any write-up.
    """
    model.eval()
    dummy = [torch.rand(3, 640, 640).to(DEVICE)]

    with torch.no_grad():
        for _ in range(10):
            model(dummy)

        times_ms = []
        for _ in range(runs):
            start = time.perf_counter()
            model(dummy)
            times_ms.append((time.perf_counter() - start) * 1000)

    mean_ms = float(np.mean(times_ms))
    p95_ms  = float(np.percentile(times_ms, 95))
    print(f"  Latency: mean={mean_ms:.1f}ms  p95={p95_ms:.1f}ms")
    return mean_ms, p95_ms


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n=== Infinity Shield O1 -- Faster R-CNN Benchmark ===")
    print(f"Device: {DEVICE}  |  torchvision {torchvision.__version__}\n")

    print("Loading dataset:")
    train_loader = DataLoader(PlasticBallDataset("train"),
                              batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn)
    valid_loader = DataLoader(PlasticBallDataset("valid"),
                              batch_size=1, collate_fn=collate_fn)
    test_loader  = DataLoader(PlasticBallDataset("test"),
                              batch_size=1, collate_fn=collate_fn)

    model     = build_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=LR,
                                momentum=0.9, weight_decay=0.0005)

    print(f"\nFine-tuning for {EPOCHS} epochs (this is slow on CPU)...")
    best_recall = -1.0
    best_path   = WEIGHTS_DIR / "fasterrcnn_plastic_v1.pt"

    for epoch in range(1, EPOCHS + 1):
        train_one_epoch(model, train_loader, optimizer, epoch)
        recall, _ = evaluate(model, valid_loader, "valid")
        if recall > best_recall:
            best_recall = recall
            torch.save(model.state_dict(), best_path)
            print(f"    -> new best (recall {recall:.3f}), saved")

    print("\nEvaluating best model on the test set:")
    model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    test_recall, test_fp_rate = evaluate(model, test_loader, "test")
    mean_ms, p95_ms = measure_latency(model)

    results = {
        "model":            "Faster R-CNN (fasterrcnn_mobilenet_v3_large_fpn)",
        "epochs":           EPOCHS,
        "device":           str(DEVICE),
        "test_recall":      round(test_recall, 4),
        "test_fp_rate":     round(test_fp_rate, 4),
        "latency_mean_ms":  round(mean_ms, 2),
        "latency_p95_ms":   round(p95_ms, 2),
    }
    out_file = RESULTS_DIR / "fasterrcnn_results.json"
    out_file.write_text(json.dumps(results, indent=2))

    print("\n=== SUMMARY ===")
    for k, v in results.items():
        print(f"  {k:18s}: {v}")
    print(f"\nWeights : {best_path}")
    print(f"Results : {out_file}")


if __name__ == "__main__":
    main()