import torch as th
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class CustomFeatureExtractor(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        n_input_channels = observation_space.shape[0]
        self.feature_extractor_net = nn.Sequential(
            nn.Linear(n_input_channels, 256),
            nn.Tanh(),
            nn.Linear(256, features_dim),
            nn.Tanh(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.feature_extractor_net(observations)
