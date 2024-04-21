"""QuadX Hover Environment."""

from __future__ import annotations

from typing import Any

import numpy as np

from PyFlyt.gym_envs.quadx_mod_envs.quadx_base_env import QuadXBaseEnv


class QuadXHoverEnv(QuadXBaseEnv):
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
        orn_conv: str = "ENU_FLU",
        start_pos: np.ndarray = np.array([[0.0, 0.0, 1.0]]),
        start_orn: np.ndarray = np.array([[0.0, 0.0, 0.0]]),
        randomize_start: bool = False,
        noisy_motors: bool = True,
        min_pwm: float = 0.05,
        max_pwm: float = 1.0,
        drone_model: str = "cf2x",
        simulate_wind: bool = False,
        flight_mode: int = 0,
        hovering_dome_size: float = 10.0,
        angle_representation: str = "quaternion",
        add_prev_actions_to_obs: bool = False,
        add_motors_state_to_obs: bool = False,
        alpha: float = 1,
        beta: float = 0.1,
        gamma: float = 1,
        delta: float = 0.1,
        normalize_actions: bool = True,
        render_mode: None | str = None,
        render_resolution: tuple[int, int] = (480, 480),
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
            orn_conv=orn_conv,
            start_pos=start_pos,
            start_orn=start_orn,
            noisy_motors=noisy_motors,
            min_pwm=min_pwm,
            max_pwm=max_pwm,
            drone_model=drone_model,
            simulate_wind=simulate_wind,
            flight_mode=flight_mode,
            angle_representation=angle_representation,
            add_prev_actions_to_obs=add_prev_actions_to_obs,
            add_motors_state_to_obs=add_motors_state_to_obs,
            normalize_actions=normalize_actions,
            render_mode=render_mode,
            render_resolution=render_resolution,
        )

        self.ang_pos = start_orn.astype(np.float32)

        """GYMNASIUM STUFF"""
        self.observation_space = self.combined_space

        """ENVIRONMENT CONSTANTS"""
        self.hovering_dome_size = hovering_dome_size
        self.randomize_start = randomize_start
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

    def reset(
        self, *, seed: None | int = None, options: None | dict[str, Any] = dict()
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """reset.

        Args:
            seed: seed to pass to the base environment.
            options: None
        """
        if self.randomize_start:
            x = np.random.uniform(
                -(self.hovering_dome_size - 1), (self.hovering_dome_size - 1)
            )
            y = np.random.uniform(
                -(self.hovering_dome_size - 1), (self.hovering_dome_size - 1)
            )
            if self.orn_conv == "ENU_FLU":
                z = np.random.uniform(1, self.hovering_dome_size - 1)
            elif self.orn_conv == "NED_FRD":
                z = np.random.uniform(-1, -(self.hovering_dome_size - 1))

            self.start_pos = np.array([[x, y, z]], dtype=np.float32).round(3)

            # Initialize the orientation
            # Initalize phi randomly between -10deg and 10deg in radians
            phi = np.random.uniform(-0.174533, 0.174533)
            # Initalize theta randomly between -10deg and 10deg in radians
            theta = np.random.uniform(-0.174533, 0.174533)
            # Initalize psi randomly between -pi/2 and 3/2*pi
            psi = np.random.uniform(-1.5708, 4.71239)

            self.start_orn = np.array([[phi, theta, psi]], dtype=np.float32).round(3)

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
        ang_vel, ang_pos, lin_vel, lin_pos, quarternion = super().compute_attitude()

        self.ang_pos = ang_pos

        ang_pos_error = np.array(
            [0, 0, np.sin(self.start_orn[0][-1])] - np.sin(ang_pos)
        )
        lin_pos_error = np.array(self.start_pos[0] - lin_pos)

        if self.add_motors_state_to_obs:
            aux_state = super().compute_auxiliary()

        # combine everything
        if self.add_motors_state_to_obs and self.add_prev_actions_to_obs:
            if self.angle_representation == 0:
                self.state = np.array(
                    [*ang_vel, *ang_pos, *lin_vel, *lin_pos, *self.action, *aux_state],
                    dtype=np.float32,
                )
            elif self.angle_representation == 1:
                self.state = np.array(
                    [
                        *ang_vel,
                        *quarternion,
                        *lin_vel,
                        *lin_pos,
                        *self.action,
                        *aux_state,
                    ],
                    dtype=np.float32,
                )
        elif self.add_motors_state_to_obs:
            if self.angle_representation == 0:
                self.state = np.array(
                    [*ang_vel, *ang_pos, *lin_vel, *lin_pos, *aux_state],
                    dtype=np.float32,
                )
            elif self.angle_representation == 1:
                self.state = np.array(
                    [*ang_vel, *quarternion, *lin_vel, *lin_pos, *aux_state],
                    dtype=np.float32,
                )
        elif self.add_prev_actions_to_obs:
            if self.angle_representation == 0:
                self.state = np.array(
                    [*ang_vel, *ang_pos, *lin_vel, *lin_pos, *self.action],
                    dtype=np.float32,
                )
            elif self.angle_representation == 1:
                self.state = np.array(
                    [*ang_vel, *quarternion, *lin_vel, *lin_pos, *self.action],
                    dtype=np.float32,
                )
        else:
            self.state = np.array(
                [*ang_vel, *ang_pos_error, *lin_vel, *lin_pos_error], dtype=np.float32
            ).round(3)

    def compute_term_trunc_reward(self):
        """Computes the termination, truncation, and reward of the current timestep."""
        super().compute_base_term_trunc_reward()

        if self.termination:
            return

        error_distance = np.linalg.norm(self.state[9:12])
        error_velocity = np.linalg.norm(self.state[6:9])
        error_orientation = np.linalg.norm(self.state[3:6])
        error_angular_velocity = np.linalg.norm(self.state[0:3])

        self.reward = 20 + (
            (-self.alpha * error_distance)
            + (-self.beta * error_velocity)
            + (-self.gamma * error_orientation)
            + (-self.delta * error_angular_velocity)
        )

        # if error_distance > self.hovering_dome_size:
        #     self.reward = 10 * self.reward
        #     return
