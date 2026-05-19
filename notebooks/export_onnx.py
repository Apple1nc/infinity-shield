from ultralytics import YOLO
model = YOLO("models/checkpoints/yolov8n_plastic_v1.pt")
model.export(format="onnx", imgsz=640, dynamic=False)