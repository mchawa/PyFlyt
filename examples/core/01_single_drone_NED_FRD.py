"""Spawn a single drone on x=0, y=0, z=1, with 0 rpy."""

import numpy as np

from PyFlyt.core import Aviary

# the starting position and orientations
start_pos = np.array([[0.0, 0.0, -1.0]])
start_orn = np.array([[0.0, 0.0, 0.0]])

drone_options = {
    "noisy_motors": False,
    "min_pwm": 0.0,
    "max_pwm": 1.0,
    "drone_model": "cf2x",
}

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

# set to position control
env.set_mode(7)

# env.set_setpoint(0, np.array([0.371, 0.371, 0.369, 0.369]))

# 90 deg in rad = 1.5708
env.set_setpoint(0, np.array([1, 1, 1.5708, -5]))

# simulate for 1000 steps (1000/120 ~= 8 seconds)
for i in range(2000):
    env.step()