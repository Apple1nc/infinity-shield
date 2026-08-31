import json
import sys, os
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'notebooks'))
from ekf_predictor import EKF

data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'trajectories.json')

with open(data_path, 'r') as f:
    data = json.load(f)

trajectory = data[0]
dt = 1 / trajectory['meta']['sample_rate']

p = trajectory['noisy'][0]
initial_state       = np.array([p[0], p[1], p[2], 0.0, 0.0, 0.0])
initial_uncertainty = np.eye(6) * 1000
measurement_noise   = np.eye(3) * (0.015 ** 2)

# load data, get one trajectory, dt from sample_rate
# build the EKF (same construction as test_ekf.py — initial state, P, Q, R)

K = 7   # points to watch before forecasting (match your ODE test)

for q in [0.01, 0.001, 0.0001, 0.00001]:
    process_noise = np.eye(6) * q
    ekf_predictor = EKF(initial_state, initial_uncertainty, dt, process_noise, measurement_noise)

    # Phase 1 — watch: feed first K points through predict/update
    for point in trajectory['noisy'][:K]:
        ekf_predictor.predict()
        ekf_predictor.update(np.array(point[:3]))

    # Phase 2 — forecast: how many steps remain?
    remaining = len(trajectory['clean']) - K
    future = ekf_predictor.predict_future(remaining)

    # Phase 3 — compare forecast against CLEAN ground truth for the future window
    # future[i] lines up with which clean point? (think: the point after the K you consumed)
    step_errors = []
    for forecast, truth in zip(future, trajectory['clean'][K:]):
        dx = forecast[0] - truth[0]
        dy = forecast[1] - truth[1]
        dz = forecast[2] - truth[2]
        distance = (dx**2 + dy**2 + dz**2) ** 0.5
        step_errors.append(distance)
        # euclidean distance, append
        # print per-step + mean

    for i, e in enumerate(step_errors):
        print(f"step {i}: {e:.4f} m")
    print(f"mean forecast error: {sum(step_errors)/len(step_errors):.4f} m")