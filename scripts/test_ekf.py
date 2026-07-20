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

for q in [0.01, 0.001, 0.0001, 0.00001]:
    process_noise = np.eye(6) * q
    ekf_predictor = EKF(initial_state, initial_uncertainty, dt, process_noise, measurement_noise)

    step_errors = []
    for noisy_point, clean_point in zip(trajectory['noisy'], trajectory['clean']):
        ekf_predictor.predict()
        ekf_predictor.update(np.array(noisy_point[:3]))
        est = ekf_predictor.state[0:3].copy()

        dx = est[0] - clean_point[0]
        dy = est[1] - clean_point[1]
        dz = est[2] - clean_point[2]
        distance = (dx**2 + dy**2 + dz**2) ** 0.5
        step_errors.append(distance)

    print(f"Q={q}:  mean error: {sum(step_errors)/len(step_errors):.4f} m")