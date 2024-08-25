"""
train.py
--------

Description:
  This file implements functions for training agents.
"""
import collections
import time

import gymnasium
import numpy as np
import torch

import agent
import performance
import utils


def create_env(env_name: str,
               num_frame_stacking: int = 1,
               image_obs: bool = False,
               n_eval_episodes: int = 100,
               record_video: bool = False) -> gymnasium.Env:
    """
    Create an environment and apply AtariProcessing wrappers if it is an image-based env.
    All environments are wrapped by the `FrameStack` wrapper.

    Args:
        env_name: Name of environment to be created.
        num_frame_stacking: Number of frames to stack.
        image_obs: Whether it is an image-based env.
        n_eval_episodes: Number of episodes to be tracked by the RecordEpisodeStatistics.
        record_video: Whether this env is used to collect frames for video creation.

    Returns:
        env: A gymnasium environment.

    """
    render_mode = "rgb_array_list" if record_video else None
    env = gymnasium.make(env_name, render_mode=render_mode)

    if image_obs:
        env = gymnasium.wrappers.AtariPreprocessing(env=env)

    env = gymnasium.wrappers.FrameStack(env, num_frame_stacking)
    env = gymnasium.wrappers.RecordEpisodeStatistics(env, deque_size=n_eval_episodes)

    return env


def train(seed: int,
          env: gymnasium.Env,
          agent: agent.Agent,
          training_steps: int,
          train_frequency: int,
          target_update_frequency: int,
          learning_starts: int,
          n_eval_episodes: int,
          evaluation_points: list,
          record_statistics_fraction: float,
          image_obs: bool,
          device: torch.device,
          verbose: bool,
          record_video: bool = False,
          save_path: str = "") -> performance.MetricsLogger:
    """
    Let the agent interact with the environment
    during a number of steps. Collect and return training metrics.
    Additionally, print progress or record video of performance.

    Args:
        seed: For reproducibility.
        env: A gymnasium environment
        agent: An Aagent instance.
        training_steps: Number of steps to train agent.
        train_frequency: Frequency of training agent, in number of steps.
        target_update_frequency: Frequency of updating target network parameters.
        learning_starts: Number of steps to play before learning starts.
        n_eval_episodes: Number of episodes used to compute evaluation statistics.
        evaluation_points: Certain steps of training at which an evaluation is done
        record_statistics_fraction: Fraction of training steps to calculate performance statistics.
        image_obs: Whether env observations are images.
        device: A device to perform computations on.
        verbose: Whether to print training progress periodically.
        record_video: Whether to make a video at each evaluation point.
        save_path: Name of video recording to save.

    Returns:
        results_buffer: Collected statistics of the agent training.
    """
    print(f"Training of Agent with seed {seed} started.")
    start_time = time.perf_counter()

    steps = 0  # global time steps for the whole training

    # --- Keeping track of some statistics that can explain agent behaviour ---
    episodes_action_values_deque = collections.deque(maxlen=n_eval_episodes)
    episodes_policy_entropy_deque = collections.deque(maxlen=n_eval_episodes)
    episodes_losses_deque = collections.deque(maxlen=n_eval_episodes)
    record_stats_frequency = int(record_statistics_fraction * training_steps)

    # fractions of training steps at which an evaluation is done
    results_buffer = performance.MetricsLogger(record_statistics_fraction, len(evaluation_points))

    frames_list = []  # list that may contain a list of frames to be used for video creation

    while steps < training_steps:
        # --- Start en episode ---
        done = False
        obs, info = env.reset(seed=seed + steps)

        action_value_sum = 0
        policy_entropy_sum = 0
        loss_sum = 0

        # --- Play an episode ---
        while not done:
            action_info = agent.act(utils.to_tensor(obs, device, image_obs, image_obs))
            action = action_info.action
            next_obs, reward, terminated, truncated, info = env.step(action)

            agent.replay_memory.push(obs, action, reward, next_obs, terminated)
            agent.decrease_epsilon()

            obs = next_obs
            done = terminated or truncated
            steps += 1

            action_value_sum += action_info.action_value
            policy_entropy_sum += action_info.entropy

            if done:
                episode_length = info["episode"]["l"]
                episodes_action_values_deque.append(action_value_sum / episode_length)
                episodes_policy_entropy_deque.append(policy_entropy_sum / episode_length)
                if loss_sum > 0:
                    episodes_losses_deque.append(loss_sum / episode_length)

            # train agent periodically
            if steps % train_frequency == 0 and steps >= learning_starts:
                loss = agent.learn()
                loss_sum += loss

            # Update the target network periodically.
            if steps % target_update_frequency == 0 and steps >= learning_starts:
                agent.update_target_network()

            # Record statistics pf past episodes.
            if steps % record_stats_frequency == 0 and steps <= training_steps:
                mean_return = np.mean(env.return_queue).round(2)
                mean_length = np.mean(env.length_queue).round()
                mean_action_value = np.mean(episodes_action_values_deque).round(2)
                mean_entropy = np.mean(episodes_policy_entropy_deque).round(2)
                mean_loss = np.nan if len(episodes_losses_deque) == 0 else np.mean(episodes_losses_deque).round(2)
                results_buffer.add(mean_return, mean_length, mean_action_value, mean_entropy, mean_loss)

                # print stats if verbose=True
                if verbose:
                    print(f"step:{steps: <10} " f"mean_episode_return={mean_return: <7.2f}  "
                          f"mean_episode_length={mean_length}", flush=True)

            # evaluate agent
            if steps in evaluation_points:
                mean_eval_return = np.mean(env.return_queue).round(2)
                results_buffer.add_evaluation_return(mean_eval_return)
                if record_video:
                    # create material for an evaluation video
                    frames = env.render()  # will return a list of last episode renderings
                    frames_list.append(frames)

    if record_video:
        # create evaluation gif
        performance.create_gif(frames_list, save_path)

    env.close()
    print(f"seed={seed}: runtime={round(time.perf_counter() - start_time, 2)}s")

    return results_buffer


def random_agent_performance(seed: int, env: gymnasium.Env, n_episodes: int) -> np.ndarray:
    """
    A random agent play, representing a baseline.
    Return episode rewards, where the number of episodes equals the number
    of times statistics are recorded for the real agent.

    Args:
        seed: Integer seed.
        env: An environment.
        n_episodes: Number of episodes to play.

    Returns:

    """
    steps = 0  # global time steps for the whole training

    for episode in range(n_episodes):
        # --- Start en episode ---
        done = False
        _, info = env.reset(seed=seed)

        # --- Play an episode ---
        while not done:
            _, _, terminated, truncated, info = env.step(env.action_space.sample())
            done = terminated or truncated

            steps += 1

    episode_returns = np.array(env.return_queue)
    env.close()

    return episode_returns


# ============== END OF FILE ==============
