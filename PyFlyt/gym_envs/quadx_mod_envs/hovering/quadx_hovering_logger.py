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

    def __init__(self, log_dir=None):
        self.log_dir = log_dir

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
            "phi (deg)",
            "theta (rad)",
            "theta (deg)",
            "psi (rad)",
            "psi (deg)",
            "p (rad/s)",
            "p (deg/s)",
            "q (rad/s)",
            "q (deg/s)",
            "r (rad/s)",
            "r (deg/s)",
            "error_x (m)",
            "error_y (m)",
            "error_z (m)",
            "error_phi (rad)",
            "error_phi (deg)",
            "error_theta (rad)",
            "error_theta (deg)",
            "error_psi (rad)",
            "error_psi (deg)",
            "motor_1_input (PWM [0-1])",
            "motor_2_input (PWM [0-1])",
            "motor_3_input (PWM [0-1])",
            "motor_4_input (PWM [0-1])",
            "reward",
        ]

    def add(self, timestamp, target_pos, target_orn, state, action, reward):
        """
        Add state action info into buffer

            Arguments:
                timestamp(int): time since episode have started
                state(tuple): environment state values
                action(int): action taken on given state

        """
        target_psi = target_orn[2]
        target_psi_deg = np.rad2deg(target_psi)

        ang_pos_rad = state[6:9]
        ang_pos_deg = np.rad2deg(ang_pos_rad)

        ang_vel_rad = state[9:12]
        ang_vel_deg = np.rad2deg(state[9:12])

        error_ang_pos_rad = state[15:18]
        error_ang_pos_deg = np.rad2deg(error_ang_pos_rad)

        entry = (
            np.concatenate(
                (
                    [timestamp],
                    target_pos,
                    [target_psi],
                    [target_psi_deg],
                    state[0:3],
                    state[3:6],
                    [ang_pos_rad[0]],
                    [ang_pos_deg[0]],
                    [ang_pos_rad[1]],
                    [ang_pos_deg[1]],
                    [ang_pos_rad[2]],
                    [ang_pos_deg[2]],
                    [ang_vel_rad[0]],
                    [ang_vel_deg[0]],
                    [ang_vel_rad[1]],
                    [ang_vel_deg[1]],
                    [ang_vel_rad[2]],
                    [ang_vel_deg[2]],
                    state[12:15],
                    [error_ang_pos_rad[0]],
                    [error_ang_pos_deg[0]],
                    [error_ang_pos_rad[1]],
                    [error_ang_pos_deg[1]],
                    [error_ang_pos_rad[2]],
                    [error_ang_pos_deg[2]],
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
        target_pos = np.array([entry[1:4] for entry in self.buffer])
        actual_pos = np.array([entry[6:9] for entry in self.buffer])
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

        # Plot the target orientation and the actual orientation in rad against time
        actual_phi_rad = np.array([entry[12] for entry in self.buffer])
        actual_theta_rad = np.array([entry[14] for entry in self.buffer])
        actual_psi_rad = np.array([entry[16] for entry in self.buffer])
        target_psi_rad = np.array([entry[4] for entry in self.buffer])
        ax1[1, 0].plot(time, actual_phi_rad, label="Actual Phi")
        ax1[1, 0].set_xlabel("Time (s)")
        ax1[1, 0].set_title("Roll Angle (rad)")
        ax1[1, 1].plot(time, actual_theta_rad, label="Actual Theta")
        ax1[1, 1].set_xlabel("Time (s)")
        ax1[1, 1].set_title("Pitch Angle (rad)")
        ax1[1, 2].plot(time, actual_psi_rad, label="Actual Psi")
        ax1[1, 2].plot(time, target_psi_rad, label="Reference Psi")
        ax1[1, 2].set_xlabel("Time (s)")
        ax1[1, 2].set_title("Yaw Angle (rad)")
        ax1[1, 2].legend()

        # Plot the target orientation and the actual orientation in deg against time
        actual_phi_deg = np.array([entry[13] for entry in self.buffer])
        actual_theta_deg = np.array([entry[15] for entry in self.buffer])
        actual_psi_deg = np.array([entry[17] for entry in self.buffer])
        target_psi_deg = np.array([entry[5] for entry in self.buffer])
        ax1[2, 0].plot(time, actual_phi_deg, label="Actual Phi")
        ax1[2, 0].set_xlabel("Time (s)")
        ax1[2, 0].set_title("Roll Angle (deg)")
        ax1[2, 1].plot(time, actual_theta_deg, label="Actual Theta")
        ax1[2, 1].set_xlabel("Time (s)")
        ax1[2, 1].set_title("Pitch Angle (deg)")
        ax1[2, 2].plot(time, actual_psi_deg, label="Actual Psi")
        ax1[2, 2].plot(time, target_psi_deg, label="Reference Psi")
        ax1[2, 2].set_xlabel("Time (s)")
        ax1[2, 2].set_title("Yaw Angle (deg)")
        ax1[2, 2].legend()

        fig2, ax2 = plt.subplots(3, 2, layout="constrained")
        # Plot the average error in position and orientation
        avg_error_x = np.mean(np.abs(np.array([entry[24] for entry in self.buffer])))
        avg_error_y = np.mean(np.abs(np.array([entry[25] for entry in self.buffer])))
        avg_error_z = np.mean(np.abs(np.array([entry[26] for entry in self.buffer])))

        max_error_x = np.max(np.abs(np.array([entry[24] for entry in self.buffer])))
        max_error_y = np.max(np.abs(np.array([entry[25] for entry in self.buffer])))
        max_error_z = np.max(np.abs(np.array([entry[26] for entry in self.buffer])))

        avg_error_phi_rad = np.mean(
            np.abs(np.array([entry[27] for entry in self.buffer]))
        )
        avg_error_phi_deg = np.mean(
            np.abs(np.array([entry[28] for entry in self.buffer]))
        )
        avg_error_theta_rad = np.mean(
            np.abs(np.array([entry[29] for entry in self.buffer]))
        )
        avg_error_theta_deg = np.mean(
            np.abs(np.array([entry[30] for entry in self.buffer]))
        )
        avg_error_psi_rad = np.mean(
            np.abs(np.array([entry[31] for entry in self.buffer]))
        )
        avg_error_psi_deg = np.mean(
            np.abs(np.array([entry[32] for entry in self.buffer]))
        )

        max_error_phi_rad = np.max(
            np.abs(np.array([entry[27] for entry in self.buffer]))
        )
        max_error_phi_deg = np.max(
            np.abs(np.array([entry[28] for entry in self.buffer]))
        )
        max_error_theta_rad = np.max(
            np.abs(np.array([entry[29] for entry in self.buffer]))
        )
        max_error_theta_deg = np.max(
            np.abs(np.array([entry[30] for entry in self.buffer]))
        )
        max_error_psi_rad = np.max(
            np.abs(np.array([entry[31] for entry in self.buffer]))
        )
        max_error_psi_deg = np.max(
            np.abs(np.array([entry[32] for entry in self.buffer]))
        )

        avg_reward = np.mean(np.array([entry[37] for entry in self.buffer]))

        labels = [
            "X-axis",
            "Y-axis",
            "Z-axis",
            "X-axis",
            "Y-axis",
            "Z-axis",
            "Roll",
            "Pitch",
            "Yaw",
            "Roll",
            "Pitch",
            "Yaw",
            "Roll",
            "Pitch",
            "Yaw",
            "Roll",
            "Pitch",
            "Yaw",
            "Average Reward",
        ]

        values = [
            avg_error_x,
            avg_error_y,
            avg_error_z,
            max_error_x,
            max_error_y,
            max_error_z,
            avg_error_phi_rad,
            avg_error_theta_rad,
            avg_error_psi_rad,
            max_error_phi_rad,
            max_error_theta_rad,
            max_error_psi_rad,
            avg_error_phi_deg,
            avg_error_theta_deg,
            avg_error_psi_deg,
            max_error_theta_deg,
            max_error_phi_deg,
            max_error_psi_deg,
            avg_reward,
        ]

        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"][0:3]

        ax2[0, 0].bar(labels[0:3], values[0:3], color=colors)
        ax2[0, 0].set_title("Avg. Error in Position (m)")
        ax2[0, 1].bar(labels[3:6], values[3:6], color=colors)
        ax2[0, 1].set_title("Max. Error in Position (m)")
        ax2[1, 0].bar(labels[6:9], values[6:9], color=colors)
        ax2[1, 0].set_title("Avg. Error in Orientation (rad)")
        ax2[1, 1].bar(labels[9:12], values[9:12], color=colors)
        ax2[1, 1].set_title("Max. Error in Orientation (rad)")
        ax2[2, 0].bar(labels[12:15], values[12:15], color=colors)
        ax2[2, 0].set_title("Avg. Error in Orientation (deg)")
        ax2[2, 1].bar(labels[15:18], values[15:18], color=colors)
        ax2[2, 1].set_title("Max. Error in Orientation (deg)")

        for ax in fig2.axes:
            ax.grid(False)
            for bars in ax.containers:
                ax.bar_label(bars, label_type="center", fmt="%.2f", fontweight="bold")

        fig2.suptitle(
            "Average Reward: {:.2f}".format(avg_reward),
            fontsize=12,
            fontweight="bold",
        )

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

            fig2.set_size_inches(18.5, 10.5)
            fig2.savefig(
                os.path.join(
                    self.log_dir, "evaluation_results_2_{}.png".format(file_index)
                ),
                dpi=800,
            )
