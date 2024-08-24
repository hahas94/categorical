import unittest

import numpy as np
import torch
import random
from agent import Agent
from utils import ActionInfo


class TestAgent(unittest.TestCase):
    def setUp(self):
        """Initialize common parameters and the Agent instance."""
        self.n_actions = 4
        self.capacity = 10
        self.batch_size = 4
        self.learning_rate = 0.001
        self.obs_shape = (4, 84, 84)  # Example for image observation
        self.epsilon_end = 0.0
        self.exploration_fraction = 0.5
        self.training_steps = 100
        self.gamma = 0.99
        self.v_min = -10
        self.v_max = 10
        self.n_atoms = 51
        self.image_obs = True
        self.n_hidden_units = 256
        self.device = torch.device('cpu')

        self.agent = Agent(
            n_actions=self.n_actions,
            capacity=self.capacity,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            obs_shape=self.obs_shape,
            epsilon_end=self.epsilon_end,
            exploration_fraction=self.exploration_fraction,
            training_steps=self.training_steps,
            gamma=self.gamma,
            v_min=self.v_min,
            v_max=self.v_max,
            n_atoms=self.n_atoms,
            image_obs=self.image_obs,
            n_hidden_units=self.n_hidden_units,
            device=self.device
        )

    def test_act(self):
        """Test the agent's action selection."""
        state = torch.zeros(1, *self.obs_shape)  # Dummy state
        action_info = self.agent.act(state)

        # Check if the returned action is within the valid range
        self.assertIsInstance(action_info, ActionInfo)
        self.assertTrue(0 <= action_info.action < self.n_actions)
        self.assertIsInstance(action_info.action_value, float)
        self.assertIsInstance(action_info.entropy, float)

    def test_decrease_epsilon(self):
        """Test the decrease_epsilon method."""
        initial_epsilon = 1
        for _ in range(0, int(self.exploration_fraction * self.training_steps / 2)):
            self.agent.decrease_epsilon()
        self.assertAlmostEqual(self.agent._epsilon, 0.5)
        for _ in range(int(self.exploration_fraction * self.training_steps / 2), self.training_steps):
            self.agent.decrease_epsilon()
        self.assertAlmostEqual(self.agent._epsilon, self.epsilon_end)

    def test_learn(self):
        """Test the learn method."""
        # Populate the replay memory with dummy transitions
        for _ in range(self.batch_size):
            self.agent.replay_memory.push(
                obs=np.random.rand(*self.obs_shape),
                action=random.randint(0, self.n_actions - 1),
                reward=random.random(),
                next_obs=np.random.rand(*self.obs_shape),
                terminal=random.choice([True, False])
            )

        # Call learn and ensure it returns a loss value
        loss = self.agent.learn()
        self.assertIsInstance(loss, float)


if __name__ == '__main__':
    unittest.main()
