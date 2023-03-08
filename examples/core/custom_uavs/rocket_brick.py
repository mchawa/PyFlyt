"""Implements a custom brick with a booster attached."""
from __future__ import annotations

import os

import numpy as np
import yaml
from pybullet_utils import bullet_client

from PyFlyt.core.abstractions import (
    Boosters,
    Camera,
    DroneClass,
    LiftingSurface,
    LiftingSurfaces,
)


class RocketBrick(DroneClass):
    """With enough thrust, you can make a brick fly."""

    def __init__(
        self,
        p: bullet_client.BulletClient,
        start_pos: np.ndarray,
        start_orn: np.ndarray,
        control_hz: int,
        physics_hz: int,
        drone_model: str = "rocket_brick",
        model_dir: None | str = os.path.dirname(os.path.realpath(__file__)),
        np_random: None | np.random.RandomState = None,
        use_camera: bool = False,
        use_gimbal: bool = False,
        camera_angle_degrees: int = 0,
        camera_FOV_degrees: int = 90,
        camera_resolution: tuple[int, int] = (128, 128),
    ):
        """Creates a UAV a brick acting as a lifting surface with a rocket attached.

        Args:
            p (bullet_client.BulletClient): p
            start_pos (np.ndarray): start_pos
            start_orn (np.ndarray): start_orn
            control_hz (int): control_hz
            physics_hz (int): physics_hz
            drone_model (str): drone_model
            model_dir (None | str): model_dir
            np_random (None | np.random.RandomState): np_random
            use_camera (bool): use_camera
            use_gimbal (bool): use_gimbal
            camera_angle_degrees (int): camera_angle_degrees
            camera_FOV_degrees (int): camera_FOV_degrees
            camera_resolution (tuple[int, int]): camera_resolution
        """
        super().__init__(
            p=p,
            start_pos=start_pos,
            start_orn=start_orn,
            control_hz=control_hz,
            physics_hz=physics_hz,
            model_dir=model_dir,
            drone_model=drone_model,
            np_random=np_random,
        )

        """Reads rocket_brick.yaml file and load UAV parameters"""
        with open(self.param_path, "rb") as f:
            # load all params from yaml
            all_params = yaml.safe_load(f)

            # all lifting surfaces
            surfaces = list()
            surfaces.append(
                LiftingSurface(
                    p=self.p,
                    physics_period=self.physics_period,
                    np_random=self.np_random,
                    uav_id=self.Id,
                    surface_id=0,
                    command_id=None,
                    command_sign=+1.0,
                    lifting_vector=np.array([0.0, 0.0, 1.0]),
                    forward_vector=np.array([0.0, 1.0, 0.0]),
                    aerofoil_params=all_params["brick_params"],
                )
            )
            self.lifting_surfaces = LiftingSurfaces(lifting_surfaces=surfaces)

            # motor
            booster_params = all_params["booster_params"]
            self.boosters = Boosters(
                p=self.p,
                physics_period=self.physics_period,
                np_random=self.np_random,
                uav_id=self.Id,
                booster_ids=np.array([1], dtype=int),
                fueltank_ids=np.array([0], dtype=int),
                total_fuel_mass=np.array([booster_params["total_fuel"]]),
                max_fuel_rate=np.array([booster_params["max_fuel_rate"]]),
                max_inertia=np.array(
                    [
                        [
                            booster_params["inertia_ixx"],
                            booster_params["inertia_iyy"],
                            booster_params["inertia_izz"],
                        ]
                    ]
                ),
                min_thrust=np.array([booster_params["min_thrust"]]),
                max_thrust=np.array([booster_params["max_thrust"]]),
                thrust_unit=np.array([[0.0, 1.0, 0.0]]),
                reignitable=np.array([booster_params["reignitable"]], dtype=bool),
                tau=np.array([booster_params["booster_tau"]]),
            )

        """ CAMERA """
        self.use_camera = use_camera
        if self.use_camera:
            self.camera = Camera(
                p=self.p,
                uav_id=self.Id,
                camera_id=0,
                use_gimbal=use_gimbal,
                camera_FOV_degrees=camera_FOV_degrees,
                camera_angle_degrees=camera_angle_degrees,
                camera_resolution=camera_resolution,
                camera_position_offset=np.array([0.0, -3.0, 1.0]),
                is_tracking_camera=True,
            )

        self.reset()

    def reset(self):
        """Resets the vehicle to the initial state."""
        self.set_mode(0)
        self.setpoint = np.zeros(2)
        self.cmd = np.zeros(2)

        self.p.resetBasePositionAndOrientation(self.Id, self.start_pos, self.start_orn)
        self.disable_artificial_damping()
        self.lifting_surfaces.reset()
        self.boosters.reset()
        self.update_state()

        if self.use_camera:
            self.rgbaImg, self.depthImg, self.segImg = self.camera.capture_image()

    def update_state(self):
        """Updates the current state of the UAV.

        This includes: ang_vel, ang_pos, lin_vel, lin_pos.
        """
        lin_pos, ang_pos = self.p.getBasePositionAndOrientation(self.Id)
        lin_vel, ang_vel = self.p.getBaseVelocity(self.Id)

        # express vels in local frame
        rotation = np.array(self.p.getMatrixFromQuaternion(ang_pos)).reshape(3, 3).T
        lin_vel = np.matmul(rotation, lin_vel)
        ang_vel = np.matmul(rotation, ang_vel)

        # ang_pos in euler form
        ang_pos = self.p.getEulerFromQuaternion(ang_pos)

        # create the state
        self.state = np.stack([ang_vel, ang_pos, lin_vel, lin_pos], axis=0)

        # update all lifting surface velocities
        self.lifting_surfaces.update_local_surface_velocities(rotation)

        # update auxiliary information
        self.aux_state = np.concatenate(
            (self.lifting_surfaces.get_states(), self.boosters.get_states())
        )

    def update_control(self):
        """Runs through controllers."""
        # the default mode
        if self.mode == 0:
            self.cmd = self.setpoint
            return

        # otherwise, check that we have a custom controller
        if self.mode not in self.registered_controllers.keys():
            raise ValueError(
                f"Don't have other modes aside from 0, received {self.mode}."
            )

        # custom controllers run if any
        self.cmd = self.instanced_controllers[self.mode].step(self.state, self.setpoint)

    def update_physics(self):
        """Updates the physics of the vehicle."""
        self.update_state()
        self.lifting_surfaces.cmd2forces(self.cmd)
        self.boosters.settings2forces(self.cmd[[0]], self.cmd[[1]])

    def update_avionics(self):
        """Updates state and control."""
        self.update_control()

        if self.use_camera:
            self.rgbaImg, self.depthImg, self.segImg = self.camera.capture_image()
