import numpy as np

def fit(recent_points, dt):
    position = recent_points[-1]
    vx = (recent_points[-1][0] - recent_points[-2][0]) / dt
    vy = (recent_points[-1][1] - recent_points[-2][1]) / dt
    vz = (recent_points[-1][2] - recent_points[-2][2]) / dt

    return np.array([position[0], position[1], position[2], vx, vy, vz])

def fit_physics(recent_points, dt):
    gravity = 9.81
    n = len(recent_points)
    t = np.array([(i - (n - 1)) * dt for i in range(n)])

    # pull each axis out of the points
    xs = np.array([p[0] for p in recent_points])
    ys = np.array([p[1] for p in recent_points])
    zs = np.array([p[2] for p in recent_points])

    vx, x0 = np.polyfit(t, xs, 1)
    vz, z0 = np.polyfit(t, zs, 1)

    ys_corrected = ys + 0.5 * (gravity) * t**2 
    vy, y0 = np.polyfit(t, ys_corrected, 1)

    # assemble state at t=0
    return np.array([x0, y0, z0, vx, vy, vz])

def integrate(state, num_of_steps, dt):
    future_points = []
    gravity = 9.81

    for i in range(num_of_steps):
        x, y, z, vx, vy, vz = state 

        new_x = x + vx * dt
        new_y = y + vy * dt
        new_z = z + vz * dt
        new_vx = vx
        new_vy = vy - gravity * dt
        new_vz = vz

        state = np.array([new_x, new_y, new_z, new_vx, new_vy, new_vz])
        future_points.append([new_x, new_y, new_z])

    return future_points

