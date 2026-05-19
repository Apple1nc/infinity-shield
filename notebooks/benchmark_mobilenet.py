"""
benchmark_mobilenet.py  --  Infinity Shield, Objective O1

Purpose:
    Fine-tune a MobileNet-SSD detector on the plastic-ball dataset and
    measure the SAME three metrics we measured for YOLOv8n, so the two
    models can be compared fairly:
        - recall              (did we find the balls that were there?)
        - false positive rate (how often did we flag something that wasn't a ball?)
        - latency             (how many milliseconds per image?)

Why this isn't an Ultralytics script:
    Ultralytics only trains YOLO models. MobileNet-SSD is a different
    architecture, so we use torchvision's built-in version of it
    ('ssdlite320_mobilenet_v3_large'). That means we have to write a
    small training loop ourselves -- there is no .train() shortcut.

Run from the project root, with the virtualenv active:
    python notebooks/benchmark_mobilenet.py
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
    ssdlite320_mobilenet_v3_large,
    SSDLite320_MobileNet_V3_Large_Weights,
)
import torchvision.transforms.functional as TF


# ============================================================================
# CONFIG  --  all the knobs in one place
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data" / "annotated"          # Roboflow YOLOv8 export
RESULTS_DIR  = PROJECT_ROOT / "results" / "o1_detection"
WEIGHTS_DIR  = PROJECT_ROOT / "models" / "checkpoints"

EPOCHS      = 50      # small dataset -> few epochs is enough
BATCH_SIZE  = 4
LR          = 0.001   # learning rate
CONF_THRESH = 0.4     # a prediction below this confidence is ignored
IOU_THRESH  = 0.4     # a prediction must overlap a real box this much to "count"

# The model expects: class 0 = background, class 1 = plastic_ball.
# So even though we only care about 1 object, NUM_CLASSES is 2.
NUM_CLASSES = 2

DEVICE = torch.device("cpu")   # laptop only -- no CUDA until hardware arrives

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
torch.manual_seed(42)          # makes runs repeatable


# ============================================================================
# DATASET  --  teaches PyTorch how to read one Roboflow image + its labels
# ============================================================================

class PlasticBallDataset(Dataset):
    """
    Reads a Roboflow YOLOv8 export.

    Expected folder layout (Roboflow's default):
        data/annotated/train/images/  + train/labels/
        data/annotated/valid/images/  + valid/labels/
        data/annotated/test/images/   + test/labels/

    Each label .txt file has one line per object:
        <class_id> <x_center> <y_center> <width> <height>
    where all four numbers are fractions of the image size (0 to 1).

    SSD does NOT understand that format -- it wants pixel corner
    coordinates [x1, y1, x2, y2]. The conversion happens below.
    """

    def __init__(self, split):
        img_dir   = DATA_DIR / split / "images"
        label_dir = DATA_DIR / split / "labels"

        if not img_dir.exists():
            raise FileNotFoundError(
                f"Could not find {img_dir}\n"
                f"Check your Roboflow export folder layout and adjust the paths."
            )

        # Pair each image with its label file
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

        # --- Load the image as a tensor in [0, 1] ---
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width = img.shape[:2]
        img_tensor = TF.to_tensor(img)        # shape: [3, H, W]

        # --- Convert YOLO labels -> pixel corner boxes ---
        boxes, labels = [], []
        if label_path.exists():
            for line in label_path.read_text().splitlines():
                parts = line.split()
                if len(parts) != 5:
                    continue
                _, cx, cy, bw, bh = map(float, parts)
                # centre+size (fraction) -> corners (pixels)
                x1 = (cx - bw / 2) * width
                y1 = (cy - bh / 2) * height
                x2 = (cx + bw / 2) * width
                y2 = (cy + bh / 2) * height
                boxes.append([x1, y1, x2, y2])
                labels.append(1)              # 1 = plastic_ball

        # torchvision needs tensors even when an image has no objects
        target = {
            "boxes":  torch.tensor(boxes,  dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.int64),
        }
        return img_tensor, target


def collate_fn(batch):
    """Images in a batch have different sizes, so we keep them as a
    list/tuple instead of stacking into one big tensor."""
    return tuple(zip(*batch))


# ============================================================================
# MODEL
# ============================================================================

def build_model():
    """
    Load MobileNet-SSD pretrained on COCO, then REPLACE its classification
    head so it predicts NUM_CLASSES instead of COCO's 91. The pretrained
    backbone is kept (it already knows useful image features); only the
    new head is learned for plastic balls -- standard transfer learning.

    We cannot pass num_classes alongside pretrained weights (the weights
    are locked to 91 classes), so the head is swapped manually after load.
    """
    from torchvision.models.detection.ssdlite import SSDLiteClassificationHead
    from torch import nn
    from functools import partial

    # 1. Load the full pretrained model (91 COCO classes)
    weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
    model   = ssdlite320_mobilenet_v3_large(weights=weights)

    # 2. Ask the model itself for the shapes the head must match:
    #    - in_channels: how many channels each feature map has
    #    - num_anchors: how many anchor boxes per location
    #    Running a dummy image through the backbone is the reliable way
    #    to read the channel counts -- they are not stored as a tidy
    #    attribute we can just look up.
    model.eval()
    with torch.no_grad():
        dummy        = torch.zeros(1, 3, 320, 320)
        features     = model.backbone(dummy)
        in_channels  = [feat.shape[1] for feat in features.values()]
    num_anchors = model.anchor_generator.num_anchors_per_location()

    # 3. Build a fresh classification head sized for OUR classes and
    #    plug it in. The box-regression head is class-agnostic -- it
    #    counts as "where is the box", not "what is it" -- so it stays.
    norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.03)
    model.head.classification_head = SSDLiteClassificationHead(
        in_channels, num_anchors, NUM_CLASSES, norm_layer
    )

    return model.to(DEVICE)


# ============================================================================
# TRAINING  --  one pass over the training images = one "epoch"
# ============================================================================

def train_one_epoch(model, loader, optimizer, epoch):
    model.train()                              # put model in "learning" mode
    running_loss = 0.0

    for images, targets in loader:
        images  = [img.to(DEVICE) for img in images]
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        # In training mode the model returns its own error ("loss").
        # A lower loss means predictions are closer to the real boxes.
        losses = sum(model(images, targets).values())

        optimizer.zero_grad()                  # clear last step's gradients
        losses.backward()                      # work out how to improve
        optimizer.step()                       # nudge the model weights

        running_loss += losses.item()

    avg_loss = running_loss / len(loader)
    print(f"  Epoch {epoch:2d}/{EPOCHS}  -  avg loss {avg_loss:.4f}")
    return avg_loss


# ============================================================================
# EVALUATION  --  measure recall and false positive rate
# ============================================================================

def iou(box_a, box_b):
    """Intersection-over-Union: how much two boxes overlap, 0 to 1."""
    x1 = max(box_a[0], box_b[0]);  y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2]);  y2 = min(box_a[3], box_b[3])
    overlap = max(0, x2 - x1) * max(0, y2 - y1)
    area_a  = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b  = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union   = area_a + area_b - overlap
    return overlap / union if union > 0 else 0.0


def evaluate(model, loader, split_name):
    """
    Counts three things across the whole split:
      TP (true positive)  - predicted a ball, a real ball was there
      FP (false positive) - predicted a ball, nothing was there
      FN (false negative) - missed a real ball
    Then:
      recall = TP / (TP + FN)   -> of all real balls, how many we caught
      FP rate = FP / (TP + FP)  -> of all our predictions, how many were wrong
    """
    model.eval()                               # put model in "predicting" mode
    tp = fp = fn = 0

    with torch.no_grad():                      # no learning here, just inference
        for images, targets in loader:
            images      = [img.to(DEVICE) for img in images]
            predictions = model(images)

            for pred, target in zip(predictions, targets):
                real_boxes = target["boxes"].numpy()

                # keep only confident predictions
                scores     = pred["scores"].cpu().numpy()
                pred_boxes = pred["boxes"].cpu().numpy()[scores >= CONF_THRESH]

                matched = set()                # real boxes already accounted for
                for pb in pred_boxes:
                    # find the best-overlapping real box not yet matched
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
# LATENCY  --  same method as validate_yolov8n.py for a fair comparison
# ============================================================================

def measure_latency(model, runs=100):
    """
    Time how long one image takes. We run 10 'warmup' passes first
    (the very first calls are always slow and would skew the average),
    then time 'runs' real passes on a fixed dummy image.
    Reports the mean and the 95th percentile (p95 = a near-worst case).
    """
    model.eval()
    dummy = [torch.rand(3, 320, 320).to(DEVICE)]   # SSD input size is 320x320

    with torch.no_grad():
        for _ in range(10):                        # warmup, not measured
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
    print("\n=== Infinity Shield O1 -- MobileNet-SSD Benchmark ===")
    print(f"Device: {DEVICE}  |  torchvision {torchvision.__version__}\n")

    # --- Load the three dataset splits ---
    print("Loading dataset:")
    train_loader = DataLoader(PlasticBallDataset("train"),
                              batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn)
    valid_loader = DataLoader(PlasticBallDataset("valid"),
                              batch_size=1, collate_fn=collate_fn)
    test_loader  = DataLoader(PlasticBallDataset("test"),
                              batch_size=1, collate_fn=collate_fn)

    # --- Build model + optimizer ---
    model = build_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=LR,
                                momentum=0.9, weight_decay=0.0005)

    # --- Train, keeping the best model by validation recall ---
    print(f"\nFine-tuning for {EPOCHS} epochs (this is slow on CPU)...")
    best_recall = -1.0
    best_path   = WEIGHTS_DIR / "mobilenet_ssd_plastic_v1.pt"

    for epoch in range(1, EPOCHS + 1):
        train_one_epoch(model, train_loader, optimizer, epoch)
        recall, _ = evaluate(model, valid_loader, "valid")
        if recall > best_recall:
            best_recall = recall
            torch.save(model.state_dict(), best_path)
            print(f"    -> new best (recall {recall:.3f}), saved")

    # --- Final scoring on the untouched test set ---
    print("\nEvaluating best model on the test set:")
    model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    test_recall, test_fp_rate = evaluate(model, test_loader, "test")
    mean_ms, p95_ms = measure_latency(model)

    # --- Save results so the O1 comparison notebook can read them ---
    results = {
        "model":          "MobileNet-SSD (ssdlite320_mobilenet_v3_large)",
        "epochs":         EPOCHS,
        "device":         str(DEVICE),
        "test_recall":    round(test_recall, 4),
        "test_fp_rate":   round(test_fp_rate, 4),
        "latency_mean_ms": round(mean_ms, 2),
        "latency_p95_ms":  round(p95_ms, 2),
    }
    out_file = RESULTS_DIR / "mobilenet_results.json"
    out_file.write_text(json.dumps(results, indent=2))

    print("\n=== SUMMARY ===")
    for k, v in results.items():
        print(f"  {k:18s}: {v}")
    print(f"\nWeights : {best_path}")
    print(f"Results : {out_file}")


if __name__ == "__main__":
    main()