"""QuadX Hover Environment."""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from PyFlyt.gym_envs.quadx_mod_envs.hovering.quadx_hovering_logger import Logger
from PyFlyt.gym_envs.quadx_mod_envs.trajectory_following_slow.quadx_base_env import (
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
        control_hz: int = 80,
        orn_conv: str = "NED_FRD",
        randomize_start: bool = True,
        start_pos: np.ndarray = np.array([[0.0, 0.0, -1.0]]),
        start_orn: np.ndarray = np.array([[0.0, 0.0, 0.0]]),
        random_trajectory: bool = True,
        waypoints: None | np.ndarray = None,
        goal_reach_distance: float = 0.3,
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
        alpha: float = 15,
        beta: float = 1,
        gamma: float = 0.2,
        draw_waypoints: bool = False,
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
        self.randomize_start = randomize_start
        self.random_trajectory = random_trajectory
        self.waypoints = waypoints
        self.goal_reach_distance = goal_reach_distance
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.draw_waypoints = draw_waypoints and (render_mode == "human")
        if self.draw_waypoints:
            file_dir = os.path.dirname(os.path.realpath(__file__))
            self.targ_obj_dir = os.path.join(file_dir, "../../../models/target.urdf")

    def reset(
        self, *, seed: None | int = None, options: None | dict[str, Any] = dict()
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """reset.

        Args:
            seed: seed to pass to the base environment.
            options: None
        """
        if self.randomize_start:
            # Initialize the position
            x = np.random.uniform(-self.flight_dome_size, self.flight_dome_size)
            y = np.random.uniform(-self.flight_dome_size, self.flight_dome_size)
            if self.orn_conv == "ENU_FLU":
                z = np.random.uniform(1, self.flight_dome_size)
            elif self.orn_conv == "NED_FRD":
                z = np.random.uniform(-1, -self.flight_dome_size)

            self.start_pos = np.array([x, y, z], dtype=np.float32, ndmin=2)

            # Initialize the orientation
            # Initalize phi randomly between -10deg and 10deg in radians
            phi = np.random.uniform(-0.174533, 0.174533)
            # Initalize theta randomly between -10deg and 10deg in radians
            theta = np.random.uniform(-0.174533, 0.174533)
            # Initalize psi randomly between -pi and pi
            psi = np.random.uniform(-np.pi, np.pi)

            self.start_orn = np.array([phi, theta, psi], dtype=np.float32, ndmin=2)
        else:
            self.start_pos = np.array(self.start_pos, dtype=np.float32, ndmin=2)
            self.start_orn = np.array(self.start_orn, dtype=np.float32, ndmin=2)

        super().begin_reset(seed, options)

        if self.random_trajectory:
            self.num_of_targets = int(
                np.clip(np.ceil(self.max_duration_seconds / 5), 2, None)
            )

            waypoints = np.zeros((self.num_of_targets, 4))

            for i in range(self.num_of_targets):
                samples = np.random.uniform(-10, 10, size=(3))
                for idx, sample in enumerate(samples):
                    if sample < 0 and sample > -1:
                        samples[idx] = -1
                    elif sample > 0 and sample < 1:
                        samples[idx] = 1
                    elif sample == 0:
                        samples[idx] = np.random.choice([-1, 1], size=(2))
                if i == 0:
                    base_point = self.start_pos[0]
                else:
                    base_point = waypoints[i - 1][0:3]

                new_waypoint = base_point + samples

                if np.abs(new_waypoint[0]) > self.flight_dome_size:
                    new_waypoint[0] = base_point[0] - samples[0]

                if np.abs(new_waypoint[1]) > self.flight_dome_size:
                    new_waypoint[1] = base_point[1] - samples[1]

                if (
                    np.abs(new_waypoint[2]) > self.flight_dome_size
                    or new_waypoint[2] > -1
                ):
                    new_waypoint[2] = base_point[2] - samples[2]
                
                target_psi = np.random.uniform(-np.pi, np.pi)

                waypoints[i][0:3] = new_waypoint
                waypoints[i][3] = target_psi

            self.waypoints = waypoints
        else:
            self.num_of_targets = self.waypoints.shape[0]

        if self.draw_waypoints:
            self.target_visual = []
            for target in self.waypoints:
                if self.orn_conv == "NED_FRD":
                    target = [target[1], target[0], -target[2]]

                self.target_visual.append(
                    self.env.loadURDF(
                        self.targ_obj_dir,
                        basePosition=target,
                        useFixedBase=True,
                        globalScaling=self.goal_reach_distance / 4.0,
                    )
                )

            for i, visual in enumerate(self.target_visual):
                self.env.changeVisualShape(
                    visual,
                    linkIndex=-1,
                    rgbaColor=(0, 1 - (i / len(self.target_visual)), 0, 1),
                )

        self.current_target_index = 0
        self.lin_pos_error = self.waypoints[0][0:3] - self.start_pos[0]
        self.orig_lin_pos_error_mag = np.linalg.norm(self.lin_pos_error)
        self.current_lin_pos_erro_mag = np.linalg.norm(self.lin_pos_error)
        self.yaw_error = ((self.waypoints[0][3] - self.start_orn[0][2]) + np.pi) % (2 * np.pi) - np.pi

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

        self.lin_pos_error = np.array(self.waypoints[self.current_target_index][0:3] - lin_pos)
        self.current_lin_pos_erro_mag = np.linalg.norm(self.lin_pos_error)

        self.yaw_error = (self.waypoints[self.current_target_index][3] - ang_pos[2] + np.pi) % (2 * np.pi) - np.pi

        # 5sec passed or target point reached
        if (self.random_trajectory and self.step_count > 0 and (self.step_count % (int(5 * self.control_hz)) == 0)) or ((not self.random_trajectory) and np.linalg.norm(self.lin_pos_error) < self.goal_reach_distance and np.linalg.norm(lin_vel) < 0.5):

            if self.current_target_index < self.num_of_targets-1:
                self.current_target_index += 1

            self.lin_pos_error = self.waypoints[self.current_target_index][0:3] - lin_pos
            self.current_lin_pos_erro_mag = np.linalg.norm(self.lin_pos_error)
            self.orig_lin_pos_error_mag = np.linalg.norm(self.lin_pos_error)

            self.yaw_error = (self.waypoints[self.current_target_index][3] - ang_pos[2] + np.pi) % (2 * np.pi) - np.pi

            if self.draw_waypoints:
                self.env.removeBody(self.target_visual[0])
                self.target_visual = self.target_visual[1:]

                # recolour
                for i, visual in enumerate(self.target_visual):
                    self.env.changeVisualShape(
                        visual,
                        linkIndex=-1,
                        rgbaColor=(0, 1 - (i / len(self.target_visual)), 0, 1),
                    )

        self.state = np.array(
            [
                *lin_pos,
                *lin_vel,
                *ang_pos,
                *ang_vel,
                *self.lin_pos_error,
                self.yaw_error,
            ],
            dtype=np.float32,
        ).round(3)

    def compute_term_trunc_reward(self):
        """Computes the termination, truncation, and reward of the current timestep."""
        super().compute_base_term_trunc_reward()

        if self.termination:
            return
        
        error_angular_velocity = np.linalg.norm(self.state[9:12])

        self.reward = (
            20
            - (self.alpha * (self.current_lin_pos_erro_mag / self.orig_lin_pos_error_mag))
            - (self.beta * self.yaw_error)
            - (self.gamma * error_angular_velocity)
        )
