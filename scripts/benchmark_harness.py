import json
import sys, os
import numpy as np
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'notebooks'))
from ekf_predictor import EKF
from ode_predictor import fit_physics, integrate 

data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'trajectories.json')

with open(data_path, 'r') as f:
    data = json.load(f)

# ---- PHASE A: accuracy across all trajectories ----
K = 7
ekf_traj_errors = []   # one mean error per trajectory
ode_traj_errors = []   # one mean error per trajectory

for trajectory in data:
    dt = 1 / trajectory['meta']['sample_rate']
    p = trajectory['noisy'][0]
    ekf = EKF(np.array([p[0],p[1],p[2],0,0,0]), np.eye(6)*1000, dt,
            np.eye(6)*0.001, np.eye(3)*(0.015**2))

    # watch first K points
    for point in trajectory['noisy'][:K]:
        ekf.predict()
        ekf.update(np.array(point[:3]))

    # Watch first K points for ODE
    fit_points = trajectory['noisy'][:K]
    state = fit_physics(fit_points, dt)

    # forecast the rest
    remaining = len(trajectory['clean']) - K
    future = ekf.predict_future(remaining)

    # Forecast the rest for ODE
    future_points = integrate(state, len(trajectory['clean']) - K, dt)

    # score forecast against clean
    step_errors = []
    for forecast, truth in zip(future, trajectory['clean'][K:]):
        d = ((forecast[0]-truth[0])**2 + (forecast[1]-truth[1])**2 + (forecast[2]-truth[2])**2)**0.5
        step_errors.append(d)

    ekf_traj_errors.append(sum(step_errors)/len(step_errors))

    # Scoring forecast against clean points ODE
    step_errors_ode = []
    for forecast, truth in zip(future_points, trajectory['clean'][K:]):
        d_ode = ((forecast[0]-truth[0])**2 + (forecast[1]-truth[1])**2 + (forecast[2]-truth[2])**2)**0.5
        step_errors_ode.append(d_ode)
    
    ode_traj_errors.append(sum(step_errors_ode)/len(step_errors_ode))


# summary across all trajectories
mean = sum(ekf_traj_errors)/len(ekf_traj_errors)
mean_ode = sum(ode_traj_errors)/len(ode_traj_errors)
std  = np.std(ekf_traj_errors)
std_ode  = np.std(ode_traj_errors)
print(f"EKF forecast error: mean {mean:.4f} m, std {std:.4f} m, over {len(ekf_traj_errors)} trajectories")
print(f"ODE forecast error: mean {mean_ode:.4f} m, std {std_ode:.4f} m, over {len(ode_traj_errors)} trajectories")

# ---- PHASE B: latency (separate, one representative trajectory) ----
trajectory = data[0]
dt = 1 / trajectory['meta']['sample_rate']
p = trajectory['noisy'][0]
ekf = EKF(np.array([p[0],p[1],p[2],0,0,0]), np.eye(6)*1000, dt,
        np.eye(6)*0.001, np.eye(3)*(0.015**2))

# warm-up
for _ in range(10):
    ekf.predict(); ekf.update(np.array(p[:3]))
# timed
N = 1000
start = time.perf_counter()
for _ in range(N):
    ekf.predict(); ekf.update(np.array(p[:3]))
per_call_ms = (time.perf_counter() - start) / N * 1000
print(f"EKF predict+update: {per_call_ms:.4f} ms per frame")

# ODE latency
# warm-up
for _ in range(10):
    state = fit_physics(trajectory['noisy'][:K], dt)
    future_points = integrate(state, len(trajectory['clean']) - K, dt)
# timed
N = 1000
start = time.perf_counter()
for _ in range(N):
    state = fit_physics(trajectory['noisy'][:K], dt)
    future_points = integrate(state, len(trajectory['clean']) - K, dt)
per_call_ms = (time.perf_counter() - start) / N * 1000
print(f"ODE predict+update: {per_call_ms:.4f} ms per frame")