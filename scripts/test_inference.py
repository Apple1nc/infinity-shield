"""
Infinity Shield — YOLOv8n Inference Test
Run this to confirm your environment is set up correctly.
Usage: python scripts/test_inference.py
"""

import sys
import time

def check_imports():
    print("=" * 55)
    print("  Infinity Shield — Environment Check")
    print("=" * 55)

    packages = {
        "torch": "PyTorch",
        "cv2": "OpenCV",
        "ultralytics": "Ultralytics YOLOv8",
        "numpy": "NumPy",
        "scipy": "SciPy",
    }

    all_ok = True
    for module, name in packages.items():
        try:
            m = __import__(module)
            version = getattr(m, "__version__", "installed")
            print(f"  ✓  {name:<25} {version}")
        except ImportError:
            print(f"  ✗  {name:<25} NOT FOUND — run: pip install {module}")
            all_ok = False

    return all_ok

def run_inference_test():
    import numpy as np
    import cv2
    from ultralytics import YOLO
    import torch

    print()
    print("  Device info")
    print(f"  ✓  PyTorch device:            {'CUDA (GPU)' if torch.cuda.is_available() else 'CPU (no GPU — expected on laptop)'}")

    print()
    print("  Downloading YOLOv8n weights (first run only ~6MB)...")
    model = YOLO("yolov8n.pt")  # downloads automatically

    print("  ✓  YOLOv8n weights loaded")

    # Create a synthetic test image (640x640 random pixels — just to test pipeline)
    test_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

    print()
    print("  Running inference on synthetic test image...")
    times = []
    for i in range(5):
        start = time.perf_counter()
        results = model(test_img, verbose=False)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    avg = sum(times) / len(times)
    print(f"  ✓  Inference ran successfully")
    print(f"  ✓  Average latency (5 runs):   {avg:.1f} ms  (CPU — Jetson/Pi will differ)")
    print()
    print("=" * 55)
    print("  All checks passed. Environment is ready.")
    print("  Next: train on your foam ball dataset.")
    print("=" * 55)

if __name__ == "__main__":
    ok = check_imports()
    if not ok:
        print()
        print("  Fix the missing packages above, then re-run this script.")
        sys.exit(1)
    print()
    run_inference_test()
