"""Spawn a single drone on x=0, y=0, z=1, with 0 rpy."""

import numpy as np
import pybullet as p

from PyFlyt.core import Aviary
from PyFlyt.gym_envs.utils.waypoint_handler import WaypointHandler

# the starting position and orientations
start_pos = np.array([[5.0, 0.0, -5.0]])
start_orn = np.array([[0.0, 0.0, 0.0]])

drone_options = {
    "noisy_motors": True,
    "min_pwm": 0.0,
    "max_pwm": 1.0,
    "drone_model": "cf2x",
}

waypoints = np.array(
    [
        [4.05, 2.94, -6.0],
        [1.55, 4.76, -7.0],
        [-1.55, 4.76, -8.0],
        [-4.05, 2.94, -9.0],
        [-5.0, 0.0, -10.0],
        [-4.05, -2.94, -9.0],
        [-1.55, -4.76, -8.0],
        [1.55, -4.76, -7.0],
        [4.05, -2.94, -6.0],
        [5.0, 0.0, -5.0],
    ]
)

waypointHandler = WaypointHandler(
    enable_render=True,
    num_targets=5,
    use_yaw_targets=False,
    goal_reach_distance=0.3,
    goal_reach_angle=0.1,
    flight_dome_size=10.0,
    np_random=np.random.default_rng(),
    waypoints=waypoints,
)

# environment setup
env = Aviary(
    orn_conv="NED_FRD",
    start_pos=start_pos,
    start_orn=start_orn,
    drone_type="quadx",
    drone_options=drone_options,
    darw_local_axis=True,
    render=True,
)

waypointHandler.reset(p=env)

first_point = waypointHandler.get_next_target()
print(first_point)
quarternion = p.getQuaternionFromEuler(env.all_states[0][1])
waypointHandler.distance_to_target(
    env.all_states[0][1], env.all_states[0][3], quarternion
)

# set to position control
env.set_mode(7)

# env.set_setpoint(0, np.array([0.371, 0.371, 0.369, 0.369]))

env.set_setpoint(0, np.array([first_point[0], first_point[1], None, first_point[2]]))

# simulate for 1000 steps (1000/120 ~= 8 seconds)
while True:
    if waypointHandler.target_reached():
        waypointHandler.advance_targets()

        if waypointHandler.all_targets_reached():
            break

        next_point = waypointHandler.get_next_target()
        env.set_setpoint(
            0, np.array([next_point[0], next_point[1], None, next_point[2]])
        )

    env.step()

    quarternion = p.getQuaternionFromEuler(env.all_states[0][1])
    waypointHandler.distance_to_target(
        env.all_states[0][1], env.all_states[0][3], quarternion
    )
