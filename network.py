"""
network.py
----------

Description:
  This file implements a neural network class based on the DQN architecture.
"""
import numpy as np
import torch
import torch.nn as nn


class Network(nn.Module):
    """
    Class implementation of the Deep-Q-Network architecture, where
    it outputs return distributions instead of action-values.
    """

    def __init__(self, obs_shape: tuple, n_actions: int, n_atoms: int, n_hidden_units: int, image_obs: bool):
        """
        Initialize the network. Expects hyperparameters object.

        Args:
            obs_shape: Shape of input observation. Examples: (1, 6), (1, 4, 84, 84)
            n_actions: Number of agent actions.
            n_atoms: Number of atoms in return distribution.
            n_hidden_units: Number of hidden units in fully-connected layer.
            image_obs: Whether the observations are images.
        """
        super().__init__()

        self._obs_shape = obs_shape
        self._n_actions = n_actions
        self._n_atoms = n_atoms

        if image_obs:
            self._convolutional = nn.Sequential(
                nn.Conv2d(4, 32, (8, 8), (4, 4), 0),
                nn.ReLU(),
                nn.Conv2d(32, 64, (4, 4), (2, 2), 0),
                nn.ReLU(),
                nn.Conv2d(64, 64, (3, 3), (1, 1), 0),
                nn.ReLU(),
                nn.Flatten(),
            )
        else:
            self._convolutional = nn.Sequential()

        in_features = self._fc_input_size()
        out_features = self._n_actions * self._n_atoms

        self._head = torch.nn.Sequential(
            nn.Linear(in_features=in_features, out_features=n_hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=n_hidden_units, out_features=out_features),
        )

    def _fc_input_size(self) -> int:
        """Compute size of input to first linear layer."""
        with torch.no_grad():
            example_obs = torch.zeros(*self._obs_shape)
            return int(np.prod(self._convolutional(example_obs).size()))

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        """Forward pass."""
        value_dists = self._head(self._convolutional(*args, **kwargs))
        return value_dists.view(-1, self._n_actions, self._n_atoms).softmax(2)


# ============== END OF FILE ==============
