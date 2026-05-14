from ultralytics import YOLO
import time, json

model = YOLO("models/checkpoints/yolov8n_plastic_v1.pt")

# Validate on test set
metrics = model.val(data="data/annotated/data.yaml", split="test")
print(f"mAP@0.5:    {metrics.box.map50:.4f}")
print(f"Precision:  {metrics.box.mp:.4f}")
print(f"Recall:     {metrics.box.mr:.4f}")

# Measure inference latency (100 runs, take mean)
import numpy as np, cv2
img = np.random.randint(0,255,(640,640,3),dtype=np.uint8)
times = []
for _ in range(100):
    t = time.perf_counter()
    model(img, verbose=False)
    times.append((time.perf_counter()-t)*1000)
print(f"Latency:    {np.mean(times):.1f}ms mean / {np.percentile(times,95):.1f}ms p95")