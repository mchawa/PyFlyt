from __future__ import annotations

import numpy as np
from pybullet_utils import bullet_client


class Motors:
    """Motors."""

    def __init__(
        self,
        p: bullet_client.BulletClient,
        physics_period: float,
        uav_id: int,
        motor_ids: list[int],
        tau: np.ndarray,
        max_rpm: np.ndarray,
        thrust_coef: np.ndarray,
        torque_coef: np.ndarray,
        noise_ratio: np.ndarray,
        np_random: np.random.RandomState,
    ):
        """Used for simulating an array of motors.

        Args:
            p (bullet_client.BulletClient): p
            physics_period (float): physics_period
            uav_id (int): uav_id
            motor_ids (list[int]): motor_ids
            tau (np.ndarray): motor ramp time constant
            max_rpm (np.ndarray): max_rpm
            thrust_coef (np.ndarray): thrust_coef
            torque_coef (np.ndarray): torque_coef
            noise_ratio (np.ndarray): noise_ratio
            np_random (np.random.RandomState): np_random
        """
        self.p = p
        self.physics_period = physics_period
        self.np_random = np_random

        # store IDs
        self.uav_id = uav_id
        self.motor_ids = motor_ids

        # get number of motors and assert shapes
        self.num_motors = len(motor_ids)
        assert tau.shape == (self.num_motors, 1)
        assert max_rpm.shape == (self.num_motors, 1)
        assert thrust_coef.shape == (self.num_motors, 3)
        assert torque_coef.shape == (self.num_motors, 3)
        assert noise_ratio.shape == (self.num_motors, 1)

        # motor constants
        self.tau = tau
        self.max_rpm = max_rpm
        self.thrust_coef = thrust_coef
        self.torque_coef = torque_coef
        self.noise_ratio = noise_ratio

    def reset(self):
        """reset_motors."""
        self.rpm = np.zeros((self.num_motors, 1))

    def pwm2forces(self, pwm):
        """pwm2forces.

        Args:
            pwm:
        """
        pwm = np.expand_dims(pwm, 1)

        # model the motor using first order ODE, y' = T/tau * (setpoint - y)
        self.rpm += (self.physics_period / self.tau) * (self.max_rpm * pwm - self.rpm)

        # noise in the motor rpms
        self.rpm += self.np_random.randn(*self.rpm.shape) * self.rpm * self.noise_ratio

        # rpm to thrust and torque
        thrust = (self.rpm**2) * self.thrust_coef
        torque = (self.rpm**2) * self.torque_coef

        # apply the forces
        for idx, thr, tor in zip(self.motor_ids, thrust, torque):
            self.p.applyExternalForce(
                self.uav_id, idx, thr, [0.0, 0.0, 0.0], self.p.LINK_FRAME
            )
            self.p.applyExternalTorque(self.uav_id, idx, tor, self.p.LINK_FRAME)