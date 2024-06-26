import numpy as np
from stable_baselines3.common.type_aliases import PolicyPredictor


class TrajectoryFollowingPIDExpert(PolicyPredictor):
    def __init__(self):
        self.target_pos = [0, 0, 0]
        self.target_psi = 0

        self.set_point = np.array(
            [
                self.target_pos[0],
                self.target_pos[1],
                self.target_psi,
                self.target_pos[2],
            ],
            ndmin=2,
        )

    def predict(
        self,
        observation,
        state=None,
        episode_start=None,
        deterministic=False,
    ):

        target_pos = observation[0][0:3] + observation[0][12:15]
        target_psi = observation[0][8] + observation[0][15]

        self.set_point = np.array(
            [target_pos[0], target_pos[1], target_psi, target_pos[2]],
            ndmin=2,
        )

        return (self.set_point, state)
