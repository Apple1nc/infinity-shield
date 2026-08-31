import sys, os, json
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__),'..','notebooks'))
from ode_predictor import fit, fit_physics, integrate

data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'trajectories.json')

with open(data_path, 'r') as f:
    data = json.load(f)

trajectory = data[0]
dt = 1 / trajectory['meta']['sample_rate']

points_used = 7
num_of_points = len(trajectory['clean'])
points_remaining = num_of_points - points_used

fit_points = trajectory['noisy'][0:points_used]

state = fit_physics(fit_points, dt)
future_points = integrate(state, points_remaining, dt)

step_errors = []

for step, (ode_point, ground_truth) in enumerate(zip(future_points, trajectory['clean'][points_used:])):
    dx = ode_point[0] - ground_truth[0]
    dy = ode_point[1] - ground_truth[1]
    dz = ode_point[2] - ground_truth[2]

    distance = (dx**2 + dy**2 + dz**2) ** 0.5
    step_errors.append(distance)

    print(f'{step}: {distance}')

print(f'Mean error: {sum(step_errors)/len(step_errors):.4f} m')