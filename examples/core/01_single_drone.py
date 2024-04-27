"""Spawn a single drone on x=0, y=0, z=1, with 0 rpy."""

import numpy as np

from PyFlyt.core import Aviary

# the starting position and orientations
start_pos = np.array([[0.0, 0.0, 1.0]])
start_orn = np.array([[0.0, 0.0, 0.0]])

drone_options = {
    "noisy_motors": False,
    "min_pwm": 0.0,
    "max_pwm": 1.0,
    "drone_model": "cf2x",
}

# environment setup
env = Aviary(
    start_pos=start_pos,
    start_orn=start_orn,
    render=True,
    drone_type="quadx",
    drone_options=drone_options,
    darw_local_axis=True,
)

# set to position control
env.set_mode(9)

# env.set_setpoint(0, np.array([1, 1, 1.5708, 5]))

env.set_setpoint(0, np.array([0.00, -0.01, 0.00, 0.37]))

# simulate for 1000 steps (1000/120 ~= 8 seconds)
for i in range(1000):
    if i == 5:
        env.set_setpoint(0, np.array([0, 0, 0, 0.365]))
        # print(env.drones[0].state.round(3))

    if i == 200:
        env.set_setpoint(0, np.array([0, 0, 0, 0.365]))
        print(env.drones[0].state.round(3))

    env.step()
