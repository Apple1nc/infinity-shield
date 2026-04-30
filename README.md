# Infinity Shield

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
