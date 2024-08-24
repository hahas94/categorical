"""
utils.py
--------

Description:
  This file implements various functions that can be used for different ad-hoc tasks.
"""
from collections import namedtuple
from typing import Union
import numpy as np
import torch
import gymnasium

# a namedtuple for the action information of an agent's act method
ActionInfo = namedtuple("ActionInfo", field_names=["action", "action_value", "entropy"])


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
