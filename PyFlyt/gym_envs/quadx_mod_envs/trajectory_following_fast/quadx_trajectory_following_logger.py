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
    """
    Logger class to log the data into csv file

        Arguments:
            log_file_path(str): path to directory where to write info

        Index 0 : Time step
        Index 1 : X position
        Index 2 : Y position
        Index 3 : Z position
        Index 4 : X velocity
        Index 5 : Y velocity
        Index 6 : Z velocity
        Index 7 : Roll angle
        Index 8 : Pitch angle
        Index 9 : Yaw angle
        Index 10 : Roll angle in degrees
        Index 11 : Pitch angle in degrees
        Index 12 : Yaw angle in degrees
        Index 13 : Roll rate
        Index 14 : Pitch rate
        Index 15 : Yaw rate
        Index 16 : Roll rate in degrees
        Index 17 : Pitch rate in degrees
        Index 18 : Yaw rate in degrees
        Index 19 : X error
        Index 20 : Y error
        Index 21 : Z error
        Index 22 : X delta_pos
        Index 23 : Y delta_pos
        Index 24 : Z delta_pos
        Index 25 : Angle difference
        Index 26 : Angle difference in degrees
        Index 27 : Motor 1 input
        Index 28 : Motor 2 input
        Index 29 : Motor 3 input
        Index 30 : Motor 4 input
        Index 31 : Reward
    """

    def __init__(self, log_dir=None):
        self.log_dir = log_dir

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
                    state[12:19],
                    [np.rad2deg(state[18])],
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
        fig1, ax1 = plt.subplots(2, 3, layout="constrained")

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

        # Plot the actual velocity against time
        actual_vel = np.array([entry[4:7] for entry in self.buffer])
        ax1[1, 0].plot(time, actual_vel[:, 0], label="Actual")
        ax1[1, 0].set_xlabel("Time (s)")
        ax1[1, 0].set_title("X-axis Velocity (m/s)")
        ax1[1, 0].legend()
        ax1[1, 1].plot(time, actual_vel[:, 1], label="Actual")
        ax1[1, 1].set_xlabel("Time (s)")
        ax1[1, 1].set_title("Y-axis Velocity (m/s)")
        ax1[1, 1].legend()
        ax1[1, 2].plot(time, actual_vel[:, 2], label="Actual")
        ax1[1, 2].set_xlabel("Time (s)")
        ax1[1, 2].set_title("Z-axis Velocity (m/s)")
        ax1[1, 2].legend()

        plt.show()

        if self.log_dir is None:
            return

        # Check if the directory exists
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # Calculate index of the new saved results
        file_index = 0
        while os.path.exists(
            os.path.join(self.log_dir, f"evaluation_results_{file_index}.csv")
        ):
            file_index += 1

        # writing to csv file
        file_name = os.path.join(
            self.log_dir, "evaluation_results_{}.csv".format(file_index)
        )
        with open(file_name, "w") as csvfile:
            # creating a csv writer object
            csvwriter = csv.writer(csvfile)

            # writing the fields
            csvwriter.writerow(self.columns)

            # writing the data rows
            csvwriter.writerows(self.buffer)

            fig1.set_size_inches(18.5, 10.5)
            fig1.savefig(
                os.path.join(
                    self.log_dir, "evaluation_results_1_{}.png".format(file_index)
                ),
                dpi=800,
            )
