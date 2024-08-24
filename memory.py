"""
memory.py
---------

Description:
  This file implements a replay memory class.
"""

import random
from typing import Any


class ReplayMemory:
    """Implements a circular replay memory object based on list storage and with random sampling."""

    def __init__(self, capacity: int, batch_size: int):
        """
        Initialize the replay memory.
        Args:
            capacity: Size of buffer.
            batch_size: Size of minibatch sample.
        """
        self._capacity = capacity
        self._batch_size = batch_size
        self._buffer: list = []
        self._index: int = 0

    def __len__(self):
        return len(self._buffer)

    def push(self, obs: Any, action: Any, reward: Any, next_obs: Any, terminal: bool) -> None:
        """
        Add a transition to the replay memory. When the buffer is full,
        the oldest transitions are replaced with new ones.

        Args:
            obs: Agent's observation
            action: Executed action.
            reward: Reward received.
            next_obs: Resulting observation.
            terminal: Whether it is terminal transition.
        """
        if len(self._buffer) < self._capacity:
            self._buffer.append(None)
        self._buffer[self._index] = (obs, action, reward, next_obs, int(terminal))
        self._index = (self._index + 1) % self._capacity

    def sample(self) -> tuple:
        """
        Sample a minibatch of transitions.

        Raises:
             ValueError: if not enough transitions exist to sample.

         Returns:
             5-tuple of obs, actions, rewards, next_obs, dones

        """
        if len(self._buffer) < self._batch_size:
            raise ValueError("Not enough transitions to sample a minibatch")

        sample = random.sample(self._buffer, self._batch_size)
        return tuple(zip(*sample))

# ============== END OF FILE ==============
