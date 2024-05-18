import csv
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

plt.style.use("default")
mpl.rcParams.update({"axes.grid": True})

parent_dir = str(Path(__file__).resolve().parent.parent.parent)


class Logger:

    def __init__(self, log_file_path=None):
        self.log_file_path = log_file_path

        self.buffer = []

        self.columns = [
            "timestep",
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
            "delta_x (m)",
            "delta_y (m)",
            "delta_z (m)",
            "angle_diff (rad)",
            "angle_diff (deg)",
            "maximum_velocity (m/s)",
            "motor_1_input (PWM [0-1])",
            "motor_2_input (PWM [0-1])",
            "motor_3_input (PWM [0-1])",
            "motor_4_input (PWM [0-1])",
            "reward",
        ]

    def add(self, timestamp, state, action, reward):
        """
        Add state action info into buffer

            Arguments:
                timestamp(int): time since episode have started
                state(tuple): environment state values
                action(int): action taken on given state

        """
        entry = (
            np.concatenate(
                (
                    [timestamp],
                    state[0:9],
                    np.rad2deg(state[6:9]),
                    state[9:12],
                    np.rad2deg(state[9:12]),
                    state[12:20],
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
        fig1, ax1 = plt.subplots(3, 3, layout="constrained")

        # Plot the target position and the actual position against time
        time = np.array([entry[0] for entry in self.buffer]) * 1 / 80
        actual_pos = np.array([entry[1:4] for entry in self.buffer])
        error_pos = np.array([entry[19:22] for entry in self.buffer])
        target_pos = actual_pos + error_pos
        ax1[0, 0].plot(time, target_pos[:, 0], label="Reference")
        ax1[0, 0].plot(time, actual_pos[:, 0], label="Actual")
        ax1[0, 0].set_xlabel("Time (s)")
        ax1[0, 0].set_title("Linear Position X (m)")
        ax1[0, 0].legend()
        ax1[0, 1].plot(time, target_pos[:, 1], label="Reference")
        ax1[0, 1].plot(time, actual_pos[:, 1], label="Actual")
        ax1[0, 1].set_xlabel("Time (s)")
        ax1[0, 1].set_title("Linear Position Y (m)")
        ax1[0, 1].legend()
        ax1[0, 2].plot(time, target_pos[:, 2], label="Refernce")
        ax1[0, 2].plot(time, actual_pos[:, 2], label="Actual")
        ax1[0, 2].set_xlabel("Time (s)")
        ax1[0, 2].set_title("Linear Position Z (m)")
        ax1[0, 2].legend()

        # Plot the target velocity and the actual velocity against time
        actual_vel = np.array([np.linalg.norm(entry[4:7]) for entry in self.buffer])
        target_vel = np.array([entry[26] for entry in self.buffer])
        ax1[1, 0].plot(time, target_vel, label="Maximum")
        ax1[1, 0].plot(time, actual_vel, label="Actual")
        ax1[1, 0].set_xlabel("Time (s)")
        ax1[1, 0].set_title("Velocity (m/s)")
        ax1[1, 0].legend()

        plt.show()

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
