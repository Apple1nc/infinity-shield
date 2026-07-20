import numpy as np

class EKF:
    def __init__(self, initial_state, initial_uncertainty, dt, process_noise, measurement_noise):
        self.state = initial_state.copy()
        self.uncertainty = initial_uncertainty.copy()
        self.dt = dt
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise

    def predict(self):
        gravity = 9.81
        x, y, z, vx, vy, vz = self.state
        new_x = x + vx * self.dt
        new_y = y + vy * self.dt
        new_z = z + vz * self.dt
        new_vx = vx
        new_vy = vy - gravity * self.dt
        new_vz = vz
        self.state = np.array([new_x, new_y, new_z, new_vx, new_vy, new_vz])
        F = np.array([
            [1, 0, 0, self.dt, 0,       0],
            [0, 1, 0, 0,       self.dt, 0],
            [0, 0, 1, 0,       0,       self.dt],
            [0, 0, 0, 1,       0,       0],
            [0, 0, 0, 0,       1,       0],
            [0, 0, 0, 0,       0,       1],
        ])
        self.uncertainty = F @ self.uncertainty @ F.T + self.process_noise
        # self.uncertainty = self.uncertainty + self.process_noise

    def update(self, measurement):
        H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ])

        innovation = measurement - H @ self.state

        S = H @ self.uncertainty @ H.T + self.measurement_noise
        K = self.uncertainty @ H.T @ np.linalg.inv(S)

        self.state = self.state + K @ innovation

        I = np.eye(6)
        self.uncertainty = (I - K @ H) @ self.uncertainty

    def predict_future(self, num_steps):
        predicted = []
        state = self.state.copy()
        gravity = 9.81

        for i in range(num_steps):
            x, y, z, vx, vy, vz = state

            new_x = x + vx * self.dt
            new_y = y + vy * self.dt
            new_z = z + vz * self.dt
            new_vx = vx
            new_vy = vy - gravity * self.dt
            new_vz = vz

            state = np.array([new_x, new_y, new_z, new_vx, new_vy, new_vz])
            predicted.append([new_x, new_y, new_z])
        
        return predicted