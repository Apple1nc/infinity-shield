import os

folders = [
    "data/raw_video",
    "data/frames",
    "data/annotated",
    "data/real_trajectories",
    "data/synthetic_trajectories",
    "models/checkpoints",
    "models/tensorrt",
    "models/onnx",
    "notebooks",
    "ros2_ws/src",
    "arduino/servo_solenoid",
    "docs/figures",
    "docs/literature_notes",
    "scripts",
    "results/o1_detection",
    "results/o2_prediction",
    "results/o3_tradeoff",
    "results/o4_evaluation",
]

files = {
    "README.md": """# Infinity Shield

**Real-Time Projectile Interception via Edge-Optimised Detection and Trajectory Prediction**

MEng Final Year Project — University of Ghana

## Project Overview
A closed-loop edge-deployed system that detects incoming thrown objects using a stereo depth camera,
predicts their trajectory in real time, and physically deflects them before they breach a defined
40cm protection zone.

## Hardware
- Edge compute: Raspberry Pi 5 8GB (or Jetson Orin Nano Super)
- Camera: Luxonis OAK-D Lite (or OAK-D)
- Actuation: Servo pan-tilt mount + deflector paddle (or pneumatic solenoid)
- MCU: Arduino Uno

## Software Stack
- ROS 2 Humble
- YOLOv8n (detection)
- ByteTrack (tracking)
- EKF / Physics ODE / LSTM (trajectory prediction)
- DepthAI SDK
- PyTorch, OpenCV, NumPy, SciPy

## Objectives
- O1: Benchmark YOLOv8n, MobileNet-SSD, NanoDet for real-time projectile detection + XAI
- O2: Compare EKF, ODE, LSTM trajectory predictors under inference budgets (5/10/20/30ms)
- O3: Latency–accuracy–interpretability tradeoff analysis on edge hardware
- O4: Closed-loop prototype — detect, predict, deflect foam projectiles before 40cm zone

## Dataset
- Roboflow project: foam_ball (version locked — see data/annotated/dataset_version.txt)
- 400+ annotated frames, 3x augmentation, 80/10/10 split

## Folder Structure
```
data/               Raw video, extracted frames, annotated dataset, trajectory logs
models/             Trained weights, ONNX exports, TensorRT engines
notebooks/          Jupyter notebooks for training, benchmarking, XAI analysis
ros2_ws/            ROS 2 workspace and nodes
arduino/            Arduino sketches for servo/solenoid control
docs/               Figures, literature notes, supervisor updates
scripts/            Utility scripts (frame extraction, evaluation, etc.)
results/            Benchmark results per objective (O1–O4)
```

## Supervisor
Prof. Robert A. Sowah
""",

    "data/annotated/dataset_version.txt": """# Dataset Version Log
# Update this every time you export a new version from Roboflow

Roboflow Project: foam_ball
Classes: foam_ball (single class)

| Version | Date | Images (raw) | Images (augmented) | Split       | Notes                        |
|---------|------|--------------|--------------------|-------------|------------------------------|
| v1      |      |              |                    | 80/10/10    | Initial annotation           |

Export format: YOLOv8
""",

    "docs/literature_notes/README.md": """# Literature Notes

One entry per paper. Format:

## [Author et al., Year] — Title
- **Area:** Detection / Tracking / Prediction / XAI / Edge AI
- **Key result:** What did they show?
- **Limitation:** What did they NOT address?
- **Relevance to this project:** Why does it matter here?

---

""",

    "results/o1_detection/README.md": "# O1 Results — Detection Benchmarks\n\nStore benchmark CSVs, saliency maps, and model comparison tables here.\n",
    "results/o2_prediction/README.md": "# O2 Results — Trajectory Predictor Benchmarks\n\nStore predictor error tables, latency logs, and explainability outputs here.\n",
    "results/o3_tradeoff/README.md": "# O3 Results — Tradeoff Analysis\n\nStore latency-accuracy-interpretability tradeoff curves and figures here.\n",
    "results/o4_evaluation/README.md": "# O4 Results — Prototype Evaluation\n\nStore 108-trial evaluation logs, deflection success rates, and video evidence here.\n",

    ".gitignore": """# Python
venv/
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
*.egg-info/
dist/
build/

# Jupyter
.ipynb_checkpoints/

# Models (large files — store externally or use Git LFS)
models/tensorrt/*.engine
models/checkpoints/*.pt
models/onnx/*.onnx

# Data (large — back up to Google Drive, not Git)
data/raw_video/
data/frames/

# VS Code
.vscode/

# OS
.DS_Store
Thumbs.db
""",

    "scripts/extract_frames.py": """\"\"\"
Frame extraction script.
Usage: python scripts/extract_frames.py --input data/raw_video/session1.mp4 --output data/frames/session1 --every 5
\"\"\"
import argparse
import subprocess
import os

def extract_frames(input_path, output_dir, every_n=5):
    os.makedirs(output_dir, exist_ok=True)
    cmd = [
        "ffmpeg", "-i", input_path,
        "-vf", f"select=not(mod(n\\,{every_n}))",
        "-vsync", "vfr",
        os.path.join(output_dir, "frame_%04d.jpg")
    ]
    print(f"Extracting every {every_n}th frame from {input_path} -> {output_dir}")
    subprocess.run(cmd, check=True)
    frames = [f for f in os.listdir(output_dir) if f.endswith('.jpg')]
    print(f"Done. {len(frames)} frames extracted.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input video file")
    parser.add_argument("--output", required=True, help="Output directory for frames")
    parser.add_argument("--every", type=int, default=5, help="Extract every Nth frame (default: 5)")
    args = parser.parse_args()
    extract_frames(args.input, args.output, args.every)
""",
}

# Create folders (with .gitkeep so Git tracks empty dirs)
for folder in folders:
    os.makedirs(folder, exist_ok=True)
    gitkeep = os.path.join(folder, ".gitkeep")
    if not os.path.exists(gitkeep):
        open(gitkeep, "w").close()

# Create files
for path, content in files.items():
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

print("✓ Folder structure created successfully.")
print("\nFolders created:")
for folder in sorted(folders):
    print(f"  {folder}/")
print("\nFiles created:")
for path in sorted(files.keys()):
    print(f"  {path}")
