"""
agent.py
--------

Description:
  This file implements an agent class, based on the Categorical-DQN algorithm.
"""

import random
import torch
from memory import ReplayMemory
from network import Network
import utils


class Agent:
    """
    Class for the Categorical-DQN (C51) agent.
    In essence, for each action, a value distribution is returned by the network,
    from which a statistic such as the mean is computed to get the action-value.
    """

    def __init__(self,
                 n_actions: int,
                 capacity: int,
                 batch_size: int,
                 learning_rate: float,
                 obs_shape: tuple,
                 epsilon_start: float,
                 epsilon_end: float,
                 exploration_fraction: float,
                 training_steps: int,
                 gamma: float,
                 v_min: int,
                 v_max: int,
                 n_atoms: int,
                 image_obs: bool,
                 n_hidden_units: int,
                 device: torch.device):
        """
        Initialize the agent class.

        Args:
            n_actions: Number of agent actions.
            capacity: Replay memory capacity.
            batch_size: Replay memory sample size.
            learning_rate: Optimizer learning rate.
            obs_shape: Env observation shape.
            epsilon_start: Start epsilon value.
            epsilon_end: Final epsilon value.
            exploration_fraction: Fraction of training steps used to explore.
            training_steps: Number of training steps.
            gamma: discount factor
            v_min: Minimum return value
            v_max: Maximum return value
            n_atoms: Number of return values between v_min and v_max
            image_obs: Whether env observation is image
            n_hidden_units: Number of agent network's fully-connected layer units.
            device: torch.device
        """
        self._n_actions = n_actions
        self._batch_size = batch_size
        self._epsilon_end = epsilon_end
        self._epsilon = epsilon_start
        self._epsilon_decay = (epsilon_start - self._epsilon_end) / (exploration_fraction * training_steps)
        self._gamma = gamma
        self._v_min = v_min
        self._v_max = v_max
        self._n_atoms = n_atoms
        self._image_obs = image_obs
        self._device = device

        self._delta = (self._v_max - self._v_min) / (self._n_atoms - 1)
        self._z = torch.linspace(self._v_min, self._v_max, self._n_atoms).to(self._device)

        self.replay_memory = ReplayMemory(capacity=capacity, batch_size=batch_size)

        self._main_network = Network(obs_shape, n_actions, n_atoms, n_hidden_units, image_obs).to(self._device)
        self._target_network = Network(obs_shape, n_actions, n_atoms, n_hidden_units, image_obs).to(self._device)
        self.update_target_network()

        self._optimizer = torch.optim.Adam(self._main_network.parameters(), lr=learning_rate, eps=0.01 / batch_size)

    def act(self, state: torch.Tensor) -> utils.ActionInfo:
        """
        Sampling action for a given state. Actions are sampled randomly during exploration.
        The action-value is the max expected value of the action value-distribution.

        Args:
            state: Current state of agent.

        Returns:
            action_info: Information namedtuple about the sampled action.
        """

        with torch.no_grad():
            value_dists = self._main_network(state)
            expected_returns = (self._z * value_dists).sum(2)

        if random.random() > self._epsilon:
            action = expected_returns.argmax(1)
            action_probs = expected_returns.softmax(0)
        else:
            action = torch.randint(high=self._n_actions, size=(1,))
            action_probs = torch.ones(self._n_actions) / self._n_actions

        action_value = expected_returns[0, action].item()
        policy_entropy = -(action_probs * torch.log(action_probs + 1e-8)).sum().item()

        action_info = utils.ActionInfo(
            action=action.item(),
            action_value=round(action_value, 2),
            entropy=round(policy_entropy, 2),
        )

        return action_info

    def decrease_epsilon(self):
        if self._epsilon > self._epsilon_end:
            self._epsilon -= self._epsilon_decay

    def update_target_network(self):
        """Updating the parameters of the target network to equal the main network's parameters."""
        self._target_network.load_state_dict(self._main_network.state_dict())

    def learn(self) -> float:
        """Learning step, updates the main network through backpropagation. Returns loss."""
        obs, actions, rewards, next_obs, terminals = self.replay_memory.sample()

        states = utils.to_tensor(array=obs, device=self._device, normalize=self._image_obs)
        actions = utils.to_tensor(array=actions, device=self._device).view(-1, 1).long()
        rewards = utils.to_tensor(array=rewards, device=self._device).view(-1, 1)
        next_states = utils.to_tensor(array=next_obs, device=self._device, normalize=self._image_obs)
        terminals = utils.to_tensor(array=terminals, device=self._device).view(-1, 1)

        # agent predictions
        value_dists = self._main_network(states)
        # gather probs for selected actions
        probs = value_dists[torch.arange(self._batch_size), actions.view(-1), :]

        # ------------------------------ Categorical algorithm ------------------------------
        #
        # Since we are dealing with value distributions and not value functions,
        # we can't minimize the loss using MSE(reward+gamma*Q_i-1 - Q_i). Instead,
        # we project the support of the target predictions T_hat*Z_i-1 onto the support
        # of the agent predictions Z_i, and minimize the cross-entropy term of
        # KL-divergence `KL(projected_T_hat*Z_i-1 || Z_i)`.
        #
        with torch.no_grad():
            # target agent predictions
            target_value_dists = self._target_network(next_states)
            target_expected_returns = (self._z * target_value_dists).sum(2)
            target_actions = target_expected_returns.argmax(1)
            target_probs = target_value_dists[torch.arange(self._batch_size), target_actions, :]

            m = torch.zeros(self._batch_size * self._n_atoms).to(self._device)

            Tz = (rewards + (1 - terminals) * self._gamma * self._z).clip(self._v_min, self._v_max)
            bj = (Tz - self._v_min) / self._delta

            l, u = torch.floor(bj).long(), torch.ceil(bj).long()

            offset = (
                torch.linspace(0,  (self._batch_size - 1) * self._n_atoms, self._batch_size)
                .long()
                .unsqueeze(1)
                .expand(self._batch_size, self._n_atoms)
                .to(self._device)
            )

            m.index_add_(
                0,
                (l + offset).view(-1),
                (target_probs * (u + (l == u).long() - bj)).view(-1).float(),
            )
            m.index_add_(
                0, (u + offset).view(-1), (target_probs * (bj - l)).view(-1).float()
            )

            m = m.view(self._batch_size, self._n_atoms)
        # -----------------------------------------------------------------------------------

        loss = (-((m * torch.log(probs + 1e-8)).sum(dim=1))).mean()

        self._optimizer.zero_grad()  # set all gradients to zero
        loss.backward()  # backpropagate loss through the network
        self._optimizer.step()  # update weights

        return round(loss.item(), 2)


# ============== END OF FILE ==============
