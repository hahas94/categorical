"""
utils.py
--------

Description:
  This file implements various functions that can be used for different ad-hoc tasks.
"""
import typing
from collections import namedtuple
import random
from typing import Union
import numpy as np
import torch
import gymnasium


# a namedtuple for the action information of an agent's act method
ActionInfo = namedtuple("ActionInfo", field_names=["action", "action_value", "entropy"])


class Hyperparameters(typing.NamedTuple):
    """Hyperparameters for Categorical-DQN."""
    # --- training ---
    n_training_steps: int  # number of steps to train agent for
    learning_starts: int  # number of steps played before agent learns.
    train_frequency: int  # how often to update agent network parameters
    target_update_frequency: int  # target network update frequency
    gamma: float  # discount factor

    # --- exploration-exploitation strategy ---
    epsilon_start: float
    epsilon_end: float
    exploration_fraction: float  # fraction of training steps to explore

    # --- neural network ---
    n_hidden_units: int  # output of first linear layer

    # --- optimizer ---
    learning_rate: float

    # --- replay memory ---
    capacity: int
    batch_size: int  # number of experiences to sample for learning step

    # --- agent algorithm ---
    v_min: int
    v_max: int
    n_atoms: int


def set_seed(seed: int):
    """Seeding libraries that use random number generators, for reproducibility purposes."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    return


def to_tensor(
    array: Union[np.ndarray, gymnasium.wrappers.LazyFrames],
    device: torch.device,
    normalize: bool = False,
    new_axis: bool = False,
) -> torch.Tensor:
    """
    Takes any array-like object and turns it into a torch.Tensor on `device`.
    The normalize parameter can be used to change the observation's range to [0,1].

    Args:
        array: An array, which can be states, actions, rewards, etc.
        device: Training device.
        normalize: Whether to normalize image observations.
        new_axis: Whether to add a new axis at the first dimension.

    Returns:
        tensor
    """
    array = np.array(array)

    if normalize:
        array = array / 255.0

    tensor = torch.tensor(array, device=device).float()

    if new_axis:
        tensor = tensor.unsqueeze(0)

    return tensor

# ============== END OF FILE ==============
