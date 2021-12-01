import copy
import numpy as np

from pybullet_swarming.environment.environment import *
from pybullet_swarming.utility.shebangs import  *
from pybullet_swarming.flier.swarm_controller import *

class Simulator():
    """
    Class wrapper around `environment` to be concise with the swarm controller
    Control is done using linear velocity setpoints and yawrate:
        vx, vy, vz, vr
    States is full linear position and yaw
        x, y, z, r
    """
    def __init__(self, start_pos, start_orn):

        # instantiate the digital twin
        self.env = Aviary(start_pos=start_pos, start_orn=start_orn, render=True)
        self.env.set_mode(6)

        # keep track of runtime
        self.steps = 0
        self.step()


    def set_setpoints(self, setpoints: np.ndarray):
        """
        setpoints is a num_drones x 4 array, where the 4 corresponds to vx, vy, vz, vr
        """
        # the setpoints in the digital twin has the last two dims flipped
        temp = copy.deepcopy(setpoints[:, -2])
        setpoints[:, -2] = copy.deepcopy(setpoints[:, -1])
        setpoints[:, -1] = temp
        self.env.set_setpoints(setpoints)


    def step(self):
        self.steps += 1
        self.env.step()


    def get_states(self):
        states = np.zeros((self.num_drones, 4))
        states[:, :-1] = copy.deepcopy(self.env.states[:, -1, :])
        states[:, -1] = copy.deepcopy(self.env.states[:, 1, -1])

        return states


    def sleep(self, seconds: float):
        for _ in range(int(seconds / self.env.period)):
            self.step()


    @property
    def states(self):
        return self.get_states()


    @property
    def num_drones(self):
        return self.env.num_drones


    @property
    def elapsed_time(self):
        return self.env.period * self.steps
