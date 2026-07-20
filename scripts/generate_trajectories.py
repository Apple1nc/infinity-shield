# What we have - starting position, starting velociy,
#                how long the throw last, how often to
#                sample points
# Variables to store - starting position
#                      starting velociy
#                      throw duration
#                      sample rate

# What happens - At each timestamp which we will get
#                using the sample rate, get the new
#                position by using the previous velocity
#                to update the position. This will include
#                using the starting to update the x, y and z
#                coordinates and using gravity to modify
#                the y coordinate. (Not too sure about this.
#                I should be because it's basic physics, but
#                I think the velocity would have different
#                components which would affects different part
#                of the position variable)
#                Next update the velocity. In the text, you 
#                mentioned that will be done using the drag and
#                gravity but thinking about it how would it be done. 

# init_pos = (0, 0, 0)
# init_vel = (10, 8, 12)
# throw_duration = 2
# sample_rate = 30
# gravity = 9.81
# trajectory = []

# def generate_one_trajectory(x0, y0, z0, vx0, vy0, vz0, throw_duration, sample_rate):
#     gravity = 9.81
#     x, y, z = x0, y0, z0
#     vx, vy, vz = vx0, vy0, vz0
#     trajectory = []
#     dt = 1/sample_rate
#     sample_num = int(throw_duration * sample_rate)

#     for i in range(sample_num): 
#         t = i/sample_rate

#         trajectory.append((x, y, z, t))  

#         x += (vx * dt)
#         y += (vy * dt)
#         z += (vz * dt)
        
#         vy -= (9.81 * dt)

#     return trajectory

import random, math, json

def generate_one_trajectory(x0, y0, z0, vx0, vy0, vz0, throw_duration, sample_rate, gravity=9.81):
    x, y, z = x0, y0, z0
    vx, vy, vz = vx0, vy0, vz0
    dt = 1 / sample_rate
    sample_num = int(throw_duration * sample_rate)
    
    trajectory = [(x, y, z, 0.0)]

    for i in range(1, sample_num + 1):
        x = x + vx * dt
        y = y + vy * dt
        z = z + vz * dt
        vy = vy - gravity * dt
        t = i * dt

        trajectory.append((x, y, z, t))

    return trajectory


def generate_dataset(num_of_trajectories):
    trajectories = []

    for i in range(num_of_trajectories):
        throw_duration = random.uniform(1.0, 1.5)
        x = random.uniform(-0.5, 0.5)
        y = random.uniform(1.2, 1.8)
        z = random.uniform(1.5, 2.5)
        v = random.uniform(7, 15)
        elevation_angle = random.uniform(5, 35)
        azimuth_angle = random.uniform(-15, 15)
        sample_rate = 30

        elevation_rad = math.radians(elevation_angle)
        azimuth_rad = math.radians(azimuth_angle)

        vy = v * math.sin(elevation_rad)
        horizontal_component = v * math.cos(elevation_rad)

        vx = horizontal_component * math.sin(azimuth_rad)
        vz = -horizontal_component * math.cos(azimuth_rad)

        trajectory = generate_one_trajectory(x, y, z, vx, vy, vz, throw_duration, sample_rate)
        # trajectories.append(trajectory)
        trajectories.append(
            {"id": i,
            "clean": trajectory,
            "meta": {
                "x0": x, "y0": y, "z0": z,
                "vx0": vx, "vy0": vy, "vz0": vz,
                "duration": throw_duration,
                "sample_rate": sample_rate
            }}
        )

    return trajectories


def add_noise(trajectory, xy_std = 0.005, z_std = 0.015):
    trajectory_noisy = []

    for point in trajectory:
        x_noisy = point[0] + random.gauss(0, xy_std)
        y_noisy = point[1] + random.gauss(0, xy_std)
        z_noisy = point[2] + random.gauss(0, z_std)

        trajectory_noisy.append((x_noisy, y_noisy, z_noisy, point[3]))

    return trajectory_noisy


if __name__ == "__main__":
    dataset = generate_dataset(200)

    for data in dataset:
        data["noisy"] = add_noise(data["clean"])
    
    with open("data/trajectories.json", "w") as f:
        json.dump(dataset, f, indent=2)