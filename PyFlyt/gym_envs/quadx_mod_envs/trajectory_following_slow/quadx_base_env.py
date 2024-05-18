"""Base PyFlyt Environment for the QuadX model using the Gymnasim API."""

from __future__ import annotations

from typing import Any

import gymnasium
import numpy as np
import pybullet as p
from gymnasium import spaces

from PyFlyt.core.aviary import Aviary
from PyFlyt.core.utils.compile_helpers import check_numpy
from PyFlyt.core.wind.gaussian_wind import GaussianWindField
from PyFlyt.gym_envs.quadx_mod_envs.hovering.quadx_hovering_logger import Logger


class QuadXBaseEnv(gymnasium.Env):
    """Base PyFlyt Environment for the QuadX model using the Gymnasium API."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        control_hz: int = 40,
        orn_conv: str = "ENU_FLU",
        start_pos: np.ndarray = np.array([[0.0, 0.0, 1.0]]),
        start_orn: np.ndarray = np.array([[0.0, 0.0, 0.0]]),
        noisy_motors: bool = True,
        min_pwm: float = 0.0,
        max_pwm: float = 1.0,
        drone_model: str = "cf2x",
        simulate_wind: bool = False,
        flight_mode: int = 0,
        flight_dome_size: float = 100,
        max_duration_seconds: float = 10.0,
        angle_representation: str = "euler",
        normalize_obs: bool = True,
        normalize_actions: bool = True,
        render_mode: None | str = None,
        render_resolution: tuple[int, int] = (480, 480),
        logger: None | Logger = None,
    ):
        """__init__.

        Args:
            orn_conv (str): orn_conv
            start_pos (np.ndarray): start_pos
            start_orn (np.ndarray): start_orn
            noisy_motors (bool): noisy_motors
            min_pwm (float): min_pwm
            max_pwm (float): max_pwm
            flight_mode (int): flight_mode
            flight_dome_size (float): flight_dome_size
            max_duration_seconds (float): max_duration_seconds
            angle_representation (str): angle_representation
            control_hz (int): control_hz
            render_mode (None | str): render_mode
            render_resolution (tuple[int, int]): render_resolution
        """
        if 240 % control_hz != 0:
            lowest = int(240 / (int(240 / control_hz) + 1))
            highest = int(240 / int(240 / control_hz))
            raise AssertionError(
                f"`control_hz` must be round denominator of 240, try {lowest} or {highest}."
            )
        self.control_hz = control_hz

        if render_mode is not None:
            assert (
                render_mode in self.metadata["render_modes"]
            ), f"Invalid render mode {render_mode}, only {self.metadata['render_modes']} allowed."
        self.render_mode = render_mode
        self.render_resolution = render_resolution
        self.logger = logger

        """GYMNASIUM STUFF"""
        # Observation space
        minimum_z_distance = None
        maximum_z_distance = None
        if orn_conv == "ENU_FLU":
            minimum_z_distance = 0
            maximum_z_distance = flight_dome_size + 30
        elif orn_conv == "NED_FRD":
            minimum_z_distance = -(flight_dome_size + 30)
            maximum_z_distance = 0

        self.obs_low = np.array(
            [
                -(flight_dome_size + 30),  # Minimum X distance
                -(flight_dome_size + 30),  # Minimum Y distance
                minimum_z_distance,  # Minimum Z distance
                -50,  # Minimum X velocity
                -50,  # Minimum Y velocity
                -50,  # Minimum Z velocity
                -np.pi,  # Minimum Phi angle
                -np.pi,  # Minimum Theta angle
                -np.pi,  # Minimum Psi angle
                -130,  # Minimum p angular velocity
                -130,  # Minimum q angular velocity
                -130,  # Minimum r angular velocity
                -20,  # Minimum X lin_pos error
                -20,  # Minimum Y lin_pos error
                -20,  # Minimum Z lin_pos error
                -np.pi,  # Minimum Psi error angle
            ]
        )
        self.obs_high = np.array(
            [
                (flight_dome_size + 30),  # Maximum X distance
                (flight_dome_size + 30),  # Maximum Y distance
                maximum_z_distance,  # Maximum Z distance
                50,  # Maximum X velocity
                50,  # Maximum Y velocity
                50,  # Maximum Z velocity
                np.pi,  # Maximum Phi angle
                np.pi,  # Maximum Theta angle
                np.pi,  # Maximum Psi angle
                130,  # Maximum p angular velocity
                130,  # Maximum q angular velocity
                130,  # Maximum r angular velocity
                20,  # Maximum X lin_pos error
                20,  # Maximum Y lin_pos error
                20,  # Maximum Z lin_pos error
                np.pi,  # Maximum Psi error angle
            ]
        )

        if normalize_obs:
            self.observation_space = spaces.Box(
                low=np.full(len(self.obs_low), -1),
                high=np.full(len(self.obs_high), 1),
                dtype=np.float32,
            )
        else:
            self.observation_space = spaces.Box(
                low=self.obs_low, high=self.obs_high, dtype=np.float32
            )

        # Action space
        if flight_mode in [-1, 8]:
            self.action_low = np.array([0, 0, 0, 0])
            self.action_high = np.array([1, 1, 1, 1])
            if normalize_actions:
                low = np.array([-1, -1, -1, -1])
                high = np.array([1, 1, 1, 1])
            else:
                low = self.action_low
                high = self.action_high
        elif flight_mode == 9:
            self.action_low = np.array([-1, -1, -1, 0])
            self.action_high = np.array([1, 1, 1, 1])
            if normalize_actions:
                low = np.array([-1, -1, -1, -1])
                high = np.array([1, 1, 1, 1])
            else:
                low = self.action_low
                high = self.action_high
        elif flight_mode == 7:
            self.action_low = np.array([-np.inf, -np.inf, -np.inf, -np.inf])
            self.action_high = np.array([np.inf, np.inf, np.inf, np.inf])
            low = self.action_low
            high = self.action_high
        else:
            raise ValueError(
                f"Invalid flight mode {flight_mode}, only -1, 7, 8, 9 allowed."
            )
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

        """ ENVIRONMENT CONSTANTS """
        self.orn_conv = orn_conv
        self.start_pos = start_pos
        self.start_orn = start_orn
        self.flight_mode = flight_mode
        self.flight_dome_size = flight_dome_size
        self.max_duration_seconds = max_duration_seconds
        self.max_steps = int(control_hz * max_duration_seconds)
        if angle_representation == "euler":
            self.angle_representation = 0
        elif angle_representation == "quaternion":
            self.angle_representation = 1
        self.min_pwm = min_pwm
        self.max_pwm = max_pwm
        self.noisy_motors = noisy_motors
        self.drone_model = drone_model
        self.simulate_wind = simulate_wind
        self.normalize_obs = normalize_obs
        self.normalize_actions = normalize_actions

    def close(self) -> None:
        """Disconnects the internal Aviary."""
        # if we already have an env, disconnect from it
        if hasattr(self, "env"):
            self.env.disconnect()

    def reset(
        self, *, seed: None | int = None, options: dict[str, Any] | None = dict()
    ) -> tuple[Any, dict[str, Any]]:
        """reset.

        Args:
            seed (None | int): seed
            options (dict[str, Any]): options

        Returns:
            tuple[Any, dict[str, Any]]:
        """
        raise NotImplementedError

    def begin_reset(
        self,
        seed: None | int = None,
        options: None | dict[str, Any] = dict(),
        drone_options: None | dict[str, Any] = dict(),
    ) -> None:
        """The first half of the reset function."""
        super().reset(seed=seed)

        # if we already have an env, disconnect from it
        if hasattr(self, "env"):
            self.env.resetSimulation()
            self.env.disconnect()

        self.step_count = 0
        self.termination = False
        self.truncation = False
        self.state = None
        self.action = np.zeros((4,))
        self.reward = 0.0
        self.info = {}
        self.info["out_of_bounds"] = False
        self.info["collision"] = False
        self.info["env_complete"] = False

        # drone options
        if drone_options is None or (type(drone_options) and len(drone_options) == 0):
            drone_options = dict()
            drone_options["noisy_motors"] = self.noisy_motors
            drone_options["min_pwm"] = self.min_pwm
            drone_options["max_pwm"] = self.max_pwm
            drone_options["drone_model"] = self.drone_model
            drone_options["control_hz"] = self.control_hz

        if self.simulate_wind:
            wind_type = GaussianWindField
            wind_options = dict()
            wind_options["orn_conv"] = self.orn_conv
        else:
            wind_type = None
            wind_options = dict()

        # init env
        self.env = Aviary(
            orn_conv=self.orn_conv,
            start_pos=self.start_pos,
            start_orn=self.start_orn,
            drone_type="quadx",
            drone_options=drone_options,
            wind_type=wind_type,
            wind_options=wind_options,
            render=self.render_mode is not None,
            darw_local_axis=(self.render_mode == "human"),
            seed=seed,
        )

        if self.render_mode is not None:
            self.camera_parameters = self.env.getDebugVisualizerCamera()

    def end_reset(
        self, seed: None | int = None, options: None | dict[str, Any] = dict()
    ) -> None:
        """The tailing half of the reset function."""
        # register all new collision bodies
        self.env.register_all_new_bodies()

        # set flight mode
        self.env.set_mode(self.flight_mode)

        # wait for env to stabilize
        # for _ in range(1):
        #     self.env.step()

        self.compute_state()

    def compute_state(self) -> None:
        """Computes the state of the QuadX."""
        raise NotImplementedError

    def compute_auxiliary(self) -> np.ndarray:
        """This returns the auxiliary state form the drone."""
        return np.round(self.env.aux_state(0), 3).astype(np.float32)

    def compute_attitude(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """state.

        This returns the base attitude for the drone.
        - ang_vel (vector of 3 values)
        - ang_pos (vector of 3 values)
        - lin_vel (vector of 3 values)
        - lin_pos (vector of 3 values)
        - quarternion (vector of 4 values)
        """
        raw_state = self.env.state(0)

        # state breakdown
        ang_vel = raw_state[0]
        ang_pos = raw_state[1]
        lin_vel = raw_state[2]
        lin_pos = raw_state[3]

        # quarternion angles
        quarternion = p.getQuaternionFromEuler(raw_state[1])

        return ang_vel, ang_pos, lin_vel, lin_pos, quarternion

    def compute_term_trunc_reward(self) -> None:
        """compute_term_trunc_reward."""
        raise NotImplementedError

    def compute_base_term_trunc_reward(self) -> None:
        """compute_base_term_trunc_reward."""

        # if self.num_targets_reached == self.num_of_targets:
        #     self.info["env_complete"] = True
        #     self.truncation |= True

        # exceed step count
        if self.step_count >= self.max_steps:
            self.info["TimeLimit.truncated"] = True
            self.truncation |= True

        # collision
        if np.any(self.env.contact_array):
            self.reward = -100
            self.info["collision"] = True
            self.termination |= True

        # linear distance error exceeding 20m
        if np.linalg.norm(self.lin_pos_error) > 20:
            self.reward = -100
            self.info["out_of_bounds"] = True
            self.termination |= True

    def step(self, action: np.ndarray) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Steps the environment.

        Args:
            action (np.ndarray): action

        Returns:
            state, reward, termination, truncation, info
        """

        old_state = self.state.copy()

        # unsqueeze the action to be usable in aviary
        if self.normalize_actions:
            # Unnormalize the action
            action = (
                ((action + 1) / 2) * (self.action_high - self.action_low)
                + self.action_low
            ).astype(np.float32)
        self.action = action.copy()

        # reset the reward and set the action
        self.reward = 0
        self.env.set_setpoint(0, action)

        # step the environment
        self.env.step()

        # compute state, reward and done
        self.compute_state()
        self.compute_term_trunc_reward()

        # Nomralize the observation
        state = None
        if self.normalize_obs:
            state = np.clip(self.state, self.obs_low, self.obs_high)
            state = (
                ((state - self.obs_low) / (self.obs_high - self.obs_low)) * 2 - 1
            ).astype(np.float32)
        else:
            state = self.state

        # increment step count
        self.step_count += 1

        # log the episode
        if self.logger is not None:
            self.logger.add(
                self.step_count - 1,
                old_state,
                action,
                self.reward,
            )

            if self.termination or self.truncation:
                self.logger.add(
                    self.step_count,
                    self.state,
                    [0, 0, 0, 0],
                    0,
                )

                self.logger.log_episode()

        # print(
        #     "State:\n \tLinear Error: X={}, Y={}, Z={}\nAction: {}\nReward: {}\n\n".format(
        #         self.state[-3], self.state[-2], self.state[-3], self.action, self.reward
        #     )
        # )

        return state, self.reward, self.termination, self.truncation, self.info

    def render(self) -> np.ndarray:
        """render."""
        check_numpy()
        assert (
            self.render_mode is not None
        ), "Please set `render_mode='human'` or `render_mode='rgb_array'` to use this function."

        _, _, rgbaImg, _, _ = self.env.getCameraImage(
            width=self.render_resolution[1],
            height=self.render_resolution[0],
            viewMatrix=self.camera_parameters[2],
            projectionMatrix=self.camera_parameters[3],
        )

        rgbaImg = np.asarray(rgbaImg).reshape(
            self.render_resolution[0], self.render_resolution[1], -1
        )

        return rgbaImg
