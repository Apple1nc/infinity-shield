from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # downloads pretrained COCO weights automatically

results = model.train(
    data="data/annotated/data.yaml",
    epochs=50,
    imgsz=640,
    batch=8,          # reduce to 4 if laptop runs out of memory
    device="cpu",
    project="runs/detect",
    name="train",
    exist_ok=True,
    patience=15,      # stop early if no improvement for 15 epochs
)
print(f"Best weights: {results.save_dir}/weights/best.pt")