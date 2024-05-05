import csv
import os
from pathlib import Path

import numpy as np

parent_dir = str(Path(__file__).resolve().parent.parent.parent)


class Logger:

    def __init__(self, log_file_path=None):
        self.log_file_path = log_file_path

        self.buffer = []

        self.columns = [
            "timestep",
            "target_x (m)",
            "target_y (m)",
            "target_z (m)",
            "target_psi (rad)",
            "target_psi (deg)",
            "x (m)",
            "y (m)",
            "z (m)",
            "x_dot (m/s)",
            "y_dot (m/s)",
            "z_dot (m/s)",
            "phi (rad)",
            "theta (rad)",
            "psi (rad)",
            "phi (deg)",
            "theta (deg)",
            "psi (deg)",
            "p (rad/s)",
            "q (rad/s)",
            "r (rad/s)",
            "p (deg/s)",
            "q (deg/s)",
            "r (deg/s)",
            "error_x (m)",
            "error_y (m)",
            "error_z (m)",
            "error_phi (rad)",
            "error_theta (rad)",
            "error_psi (rad)",
            "error_phi (deg)",
            "error_theta (deg)",
            "error_psi (deg)",
            "motor_1_input (PWM [0-1])",
            "motor_2_input (PWM [0-1])",
            "motor_3_input (PWM [0-1])",
            "motor_4_input (PWM [0-1])",
            "reward",
        ]

    def add(self, timestamp, target_pos, target_psi, state, action, reward):
        """
        Add state action info into buffer

            Arguments:
                timestamp(int): time since episode have started
                state(tuple): environment state values
                action(int): action taken on given state

        """
        target_psi_deg = np.rad2deg(target_psi)

        ang_pos_rad = np.arcsin(state[6:9])
        ang_pos_deg = np.rad2deg(ang_pos_rad)

        ang_vel_deg = np.rad2deg(state[9:12])

        error_ang_pos_rad = np.arcsin(state[15:18])
        error_ang_pos_deg = np.rad2deg(error_ang_pos_rad)

        entry = (
            np.concatenate(
                (
                    [timestamp],
                    target_pos[0],
                    [target_psi],
                    [target_psi_deg],
                    state[0:3],
                    state[3:6],
                    state[6:9],
                    ang_pos_rad,
                    ang_pos_deg,
                    state[9:12],
                    ang_vel_deg,
                    state[12:15],
                    state[15:18],
                    error_ang_pos_rad,
                    error_ang_pos_deg,
                    action,
                    [reward],
                )
            )
            .round(3)
            .tolist()
        )
        self.buffer.append(entry)

    def empty_buffer(self):
        """
        Empty data buffer
        """
        self.buffer = []

    def log_episode(self):
        """
        Write episode log into specified file

            Arguments:
                path_to_log(str): path to directory where to write info
                name(str): name for accumulated log info

        """

        if self.log_file_path is None:
            return

        # writing to csv file
        with open(self.log_file_path, "w") as csvfile:
            # creating a csv writer object
            csvwriter = csv.writer(csvfile)

            # writing the fields
            csvwriter.writerow(self.columns)

            # writing the data rows
            csvwriter.writerows(self.buffer)
