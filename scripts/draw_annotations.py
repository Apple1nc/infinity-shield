"""
draw_annotations.py  --  Infinity Shield helper

Draws bounding boxes from YOLO-format .txt label files onto their
matching images, so you can inspect your annotations locally without
the Roboflow web interface.

Goes through ONE split (train / valid / test) and writes annotated
copies to a separate folder you can browse in Windows Explorer.

Useful for spot-checking annotation conventions -- e.g. confirming
whether held balls were boxed or not.

Run from project root, virtualenv active:
    python scripts/draw_annotations.py train
    python scripts/draw_annotations.py valid
    python scripts/draw_annotations.py test
"""

import sys
from pathlib import Path

import cv2


# ============================================================================
# CONFIG
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data" / "annotated"
OUTPUT_BASE  = PROJECT_ROOT / "data" / "annotated_visual"

BOX_COLOR  = (0, 255, 0)     # green, BGR
BOX_THICK  = 2
TEXT_COLOR = (0, 255, 0)
FONT       = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6


# ============================================================================
# MAIN
# ============================================================================

def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("train", "valid", "test"):
        print("Usage: python scripts/draw_annotations.py [train|valid|test]")
        sys.exit(1)

    split = sys.argv[1]
    img_dir   = DATA_DIR / split / "images"
    label_dir = DATA_DIR / split / "labels"
    out_dir   = OUTPUT_BASE / split

    if not img_dir.exists():
        print(f"[ERROR] Image folder not found: {img_dir}")
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nReading from : {img_dir}")
    print(f"Writing to   : {out_dir}\n")

    image_paths = sorted([p for p in img_dir.iterdir()
                          if p.suffix.lower() in (".jpg", ".jpeg", ".png")])

    if not image_paths:
        print(f"[ERROR] No images found in {img_dir}")
        sys.exit(1)

    annotated_count = 0
    boxes_total = 0

    for i, img_path in enumerate(image_paths, 1):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [skip] could not read {img_path.name}")
            continue

        h, w = img.shape[:2]
        label_path = label_dir / (img_path.stem + ".txt")

        # Read YOLO labels: class_id  cx  cy  width  height  (all 0-1)
        n_boxes = 0
        if label_path.exists():
            for line in label_path.read_text().splitlines():
                parts = line.split()
                if len(parts) != 5:
                    continue
                _, cx, cy, bw, bh = map(float, parts)
                x1 = int((cx - bw / 2) * w)
                y1 = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w)
                y2 = int((cy + bh / 2) * h)

                cv2.rectangle(img, (x1, y1), (x2, y2), BOX_COLOR, BOX_THICK)
                cv2.putText(img, "ball", (x1, max(y1 - 5, 12)),
                            FONT, FONT_SCALE, TEXT_COLOR, 2)
                n_boxes += 1

        # Burn "N boxes" + filename into the top of the image for fast scanning
        info = f"{n_boxes} box(es)  -  {img_path.name}"
        cv2.rectangle(img, (0, 0), (w, 28), (0, 0, 0), -1)
        cv2.putText(img, info, (10, 20), FONT, 0.55, (255, 255, 255), 1)

        cv2.imwrite(str(out_dir / img_path.name), img)
        annotated_count += 1
        boxes_total += n_boxes

        if i % 100 == 0:
            print(f"  processed {i}/{len(image_paths)}")

    print(f"\n[Done] {annotated_count} images written to: {out_dir}")
    print(f"       {boxes_total} boxes drawn in total.")
    print(f"\nOpen the folder in Windows Explorer and scroll through to inspect.")
    print(f"Sort by filename, or use the preview pane.")


if __name__ == "__main__":
    main()
