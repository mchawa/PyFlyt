"""QuadX Hover Environment."""

from __future__ import annotations

from typing import Any

import numpy as np

from PyFlyt.gym_envs.quadx_mod_envs.hovering.quadx_hovering_logger import Logger
from PyFlyt.gym_envs.quadx_mod_envs.trajectory_following.quadx_base_env import (
    QuadXBaseEnv,
)


class QuadXTrajectoryFollowingrEnv(QuadXBaseEnv):
    """Simple Hover Environment.

    Actions are vp, vq, vr, T, ie: angular rates and thrust.
    The target is to not crash for the longest time possible.

    Args:
        sparse_reward (bool): whether to use sparse rewards or not.
        flight_mode (int): the flight mode of the UAV
        flight_dome_size (float): size of the allowable flying area.
        max_duration_seconds (float): maximum simulation time of the environment.
        angle_representation (str): can be "euler" or "quaternion".
        agent_hz (int): looprate of the agent to environment interaction.
        render_mode (None | str): can be "human" or None.
        render_resolution (tuple[int, int]): render_resolution.
    """

    def __init__(
        self,
        control_hz: int = 40,
        orn_conv: str = "NED_FRD",
        randomize_start: bool = True,
        start_pos: np.ndarray = np.array([[0.0, 0.0, -1.0]]),
        start_orn: np.ndarray = np.array([[0.0, 0.0, 0.0]]),
        target_pos: np.ndarray = np.array([0.0, 0.0, -1.0]),
        next_pos: np.ndarray = np.array([0.0, 0.0, -1.0]),
        maximum_velocity: float = 10.0,
        min_pwm: float = 0.0,
        max_pwm: float = 1.0,
        noisy_motors: bool = False,
        drone_model: str = "cf2x",
        flight_mode: int = 9,
        simulate_wind: bool = False,
        flight_dome_size: float = 100,
        max_duration_seconds: float = 10.0,
        angle_representation: str = "euler",
        normalize_obs: bool = True,
        normalize_actions: bool = True,
        alpha: float = 10,
        beta: float = 1,
        gamma: float = 0.1,
        delta: float = 1,
        render_mode: None | str = None,
        render_resolution: tuple[int, int] = (480, 480),
        logger: None | Logger = None,
    ):
        """__init__.

        Args:
            sparse_reward (bool): whether to use sparse rewards or not.
            flight_mode (int): the flight mode of the UAV
            flight_dome_size (float): size of the allowable flying area.
            max_duration_seconds (float): maximum simulation time of the environment.
            angle_representation (str): can be "euler" or "quaternion".
            agent_hz (int): looprate of the agent to environment interaction.
            render_mode (None | str): can be "human" or None.
            render_resolution (tuple[int, int]): render_resolution.
        """
        super().__init__(
            control_hz=control_hz,
            orn_conv=orn_conv,
            start_pos=start_pos,
            start_orn=start_orn,
            min_pwm=min_pwm,
            max_pwm=max_pwm,
            noisy_motors=noisy_motors,
            drone_model=drone_model,
            flight_mode=flight_mode,
            simulate_wind=simulate_wind,
            flight_dome_size=flight_dome_size,
            max_duration_seconds=max_duration_seconds,
            angle_representation=angle_representation,
            normalize_obs=normalize_obs,
            normalize_actions=normalize_actions,
            render_mode=render_mode,
            render_resolution=render_resolution,
            logger=logger,
        )
        """ENVIRONMENT CONSTANTS"""
        self.target_pos = np.array(target_pos, dtype=np.float32).round(3)
        self.next_pos = np.array(next_pos, dtype=np.float32).round(3)
        self.delta_pos = (self.next_pos - self.target_pos).round(3)
        self.maximum_velocity = maximum_velocity
        self.randomize_start = randomize_start
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.angle_diff = 0.0
        self.last_error_distance = np.linalg.norm(self.target_pos - start_pos[0])

    def reset(
        self, *, seed: None | int = None, options: None | dict[str, Any] = dict()
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """reset.

        Args:
            seed: seed to pass to the base environment.
            options: None
        """
        if self.randomize_start:
            self.maximum_velocity = np.random.uniform(1, 20)

            # Initialize the position
            x = np.random.uniform(-self.flight_dome_size, self.flight_dome_size)
            y = np.random.uniform(-self.flight_dome_size, self.flight_dome_size)
            if self.orn_conv == "ENU_FLU":
                z = np.random.uniform(1, self.flight_dome_size)
            elif self.orn_conv == "NED_FRD":
                z = np.random.uniform(-1, -self.flight_dome_size)

            self.start_pos = np.array([x, y, z], dtype=np.float32, ndmin=2).round(3)
            target_pos_delta = np.zeros(3)
            next_pos_delta = np.zeros(3)
            for i in range(3):
                samples = np.random.uniform(-10, 10, size=(2))
                for idx, sample in enumerate(samples):
                    if sample < 0 and sample > -1:
                        samples[idx] = -1
                    elif sample > 0 and sample < 1:
                        samples[idx] = 1
                    elif sample == 0:
                        samples[idx] = np.random.choice([-1, 1], size=(2))
                target_pos_delta[i] = samples[0]
                next_pos_delta[i] = samples[1]

            self.target_pos = (self.start_pos[0] + target_pos_delta).round(3)
            self.next_pos = (self.target_pos + next_pos_delta).round(3)
            self.delta_pos = (self.next_pos - self.target_pos).round(3)

            # Initialize the orientation
            # Initalize phi randomly between -10deg and 10deg in radians
            phi = np.random.uniform(-0.174533, 0.174533)
            # Initalize theta randomly between -10deg and 10deg in radians
            theta = np.random.uniform(-0.174533, 0.174533)
            # Initalize psi randomly between -pi and pi
            psi = np.random.uniform(-np.pi, np.pi)

            self.start_orn = np.array(
                [phi, theta, psi], dtype=np.float32, ndmin=2
            ).round(3)
            self.last_error_distance = np.linalg.norm(
                self.target_pos - self.start_pos[0]
            )
            self.angle_diff = 0.0

        super().begin_reset(seed, options)
        super().end_reset(seed, options)

        return self.state, self.info

    def compute_state(self):
        """Computes the state of the current timestep.

        This returns the observation.
        - ang_vel (vector of 3 values)
        - ang_pos (vector of 3/4 values)
        - lin_vel (vector of 3 values)
        - lin_pos (vector of 3 values)
        - previous_action (vector of 4 values)
        - auxiliary information (vector of 4 values)
        """
        ang_vel, ang_pos, lin_vel, lin_pos, _ = super().compute_attitude()

        ang_pos = (ang_pos + np.pi) % (2 * np.pi) - np.pi

        lin_pos_error = np.array(self.target_pos - lin_pos)

        if self.state is not None:
            self.last_error_distance = np.linalg.norm(self.state[12:15])

        # target point reached
        if np.linalg.norm(lin_pos_error) < 0.3:
            self.target_pos = self.next_pos

            next_pos_delta = np.zeros(3)
            for i in range(3):
                sample = np.random.uniform(-10, 10)
                if sample < 0 and sample > -1:
                    sample = -1
                elif sample > 0 and sample < 1:
                    sample = 1
                elif sample == 0:
                    sample = np.random.choice([-1, 1])
                next_pos_delta[i] = sample
            self.next_pos = (self.target_pos + next_pos_delta).round(3)
            self.next_pos[0:1] = np.clip(
                self.next_pos[0:1], -self.flight_dome_size, self.flight_dome_size
            )
            self.next_pos[2] = np.clip(self.next_pos[2], -self.flight_dome_size, -1)

            self.delta_pos = (self.next_pos - self.target_pos).round(3)

            lin_pos_error = np.array(self.target_pos - lin_pos)

            self.last_error_distance = np.linalg.norm(lin_pos_error)

            self.maximum_velocity = np.random.uniform(1, 20)

        if np.linalg.norm(lin_vel) >= 0.01:
            self.angle_diff = np.arccos(
                np.dot(lin_vel, self.delta_pos)
                / (np.linalg.norm(lin_vel) * np.linalg.norm(self.delta_pos))
            )

        self.state = np.array(
            [
                *lin_pos,
                *lin_vel,
                *ang_pos,
                *ang_vel,
                *lin_pos_error,
                self.angle_diff,
                self.maximum_velocity,
            ],
            dtype=np.float32,
        ).round(3)

    def compute_term_trunc_reward(self):
        """Computes the termination, truncation, and reward of the current timestep."""
        super().compute_base_term_trunc_reward()

        if self.termination:
            return

        error_distance = np.linalg.norm(self.state[12:15])
        error_angle_diff = np.exp(-error_distance) * self.angle_diff
        error_angular_velocity = np.linalg.norm(self.state[9:12])
        error_linear_velocity = np.clip(
            np.linalg.norm(self.state[3:6]) - self.maximum_velocity, 0, None
        )

        r1 = np.clip((self.last_error_distance - error_distance) * (self.control_hz / self.maximum_velocity), None, 1)

        self.reward = (
            (self.alpha * r1)
            - (self.beta * error_angle_diff)
            - (self.gamma * error_angular_velocity)
            - (self.delta * error_linear_velocity)
        )
