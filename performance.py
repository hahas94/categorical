"""
performance.py
--------------

Description:
  This file implements functionality for storing agent performance and visualizing it.
"""
from typing import List, Tuple

import matplotlib.pyplot as plt
import moviepy.editor
import numpy as np


class MetricsLogger:
    """Logger that stores various episodic metrics."""

    def __init__(self, record_statistics_fraction: float, n_eval_points: int):
        """
        Initialize logger.

        Args:
            record_statistics_fraction: fraction of training steps to record past episodic statistics
            n_eval_points: number of time to evaluate agent
        """
        num_stats = int(1 / record_statistics_fraction)
        self.episode_returns = np.empty(num_stats)
        self.episode_lengths = np.empty(num_stats)
        self.evaluation_returns = np.empty(n_eval_points)
        self.episode_action_values = np.empty(num_stats)
        self.losses = np.empty(num_stats)
        self.policy_entropy = np.empty(num_stats)

        self.index = 0
        self.eval_index = 0

    def add(
        self,
        episode_return: float,
        episode_length: float,
        episode_action_value: float,
        entropy: float,
        loss: float,
    ):
        """
        Add episode stats.

        Args:
            episode_return: Mean episodic return of past n_eval episodes.
            episode_length: Mean episodic length of past n_eval episodes.
            episode_action_value: Mean predicted action-value of past n_eval episodes.
            entropy: Mean policy entropy of past n_eval episodes.
            loss: Mean loss of past n_eval episodes.

        Returns:

        """
        self.episode_returns[self.index] = episode_return
        self.episode_lengths[self.index] = episode_length
        self.episode_action_values[self.index] = episode_action_value
        self.policy_entropy[self.index] = entropy
        self.losses[self.index] = loss

        self.index += 1

    def add_evaluation_return(self, mean_eval_return: float):
        """Add mean evaluation return obtained at each evaluation point."""
        self.evaluation_returns[self.eval_index] = mean_eval_return
        self.eval_index += 1


def aggregate_results(lst: List[np.ndarray]) -> (np.ndarray, np.ndarray):
    """
    Aggregate a list of arrays to compute and return their mean and standard deviation.

    Args:
        lst: A list of arrays.

    Returns:
        mean, stddev: the mean and standard deviation of the arrays.

    """
    mean = np.mean(lst, axis=0).round(2)
    stddev = np.std(lst, axis=0).round(2)

    return mean, stddev


def preprocess_results(results: List[MetricsLogger]) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Combine data for various metrics and aggregate them across agents. Return the processed data."""
    stats = [
        [res_buffer.episode_returns for res_buffer in results],
        [res_buffer.episode_lengths for res_buffer in results],
        [res_buffer.episode_action_values for res_buffer in results],
        [res_buffer.losses for res_buffer in results],
        [res_buffer.policy_entropy for res_buffer in results],
        [res_buffer.evaluation_returns for res_buffer in results],
    ]

    aggregated_data = [aggregate_results(lst) for lst in stats]

    return aggregated_data


def visualize_performance(processed_data: List[Tuple[np.ndarray, np.ndarray]],
                          training_steps: int,
                          n_episodes_recorded: int,
                          evaluation_points: list,
                          baseline_returns: np.ndarray,
                          save_fig: bool = False,
                          save_path: str = ""):
    """
    Visualize the aggregated metrics collected by the agent(s).
    Additionally, the figure can be saved to disk.

    Args:
        processed_data: A list containing tuples of (mean, stddev) for each metric.
        training_steps: Number of steps of agent training.
        n_episodes_recorded: Number of episodes where agent performance is recorded at.
        evaluation_points: Points training steps at which agent is evaluated.
        baseline_returns: Array of the random baseline episodic returns.
        save_fig: Whether to save the figure.
        save_path: Path to save the figure to.

    Returns:

    """
    plt.style.use("seaborn")
    x = np.linspace((1/n_episodes_recorded) * training_steps, training_steps, n_episodes_recorded)
    color = "royalblue"
    y_labels = [
        "Return",
        "Episode Length",
        "Predicted action-value",
        "Loss",
        "Entropy",
    ]
    titles = [
        "Aggregated agents returns vs baseline",
        "Aggregated episode lengths",
        "Aggregated action-value per episode",
        "Aggregated training losses",
        "Aggregated policy entropy",
    ]

    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(14, 21))

    eval_axes: plt.Axes = axes.flatten()[-1]
    axes = axes.flatten()[:-1]

    for i, ax in enumerate(axes):
        mean, std = processed_data[i]
        ax.plot(x, mean, color=color, label="mean")
        ax.fill_between(x=x, y1=mean - std, y2=mean + std, label="stddev", alpha=0.2, color="tomato")
        ax.set(xlabel="Steps", ylabel=y_labels[i], title=titles[i])
        ax.legend()

    axes[0].plot(x, baseline_returns, color="black", label="baseline")
    axes[0].legend()

    eval_mean, eval_std = processed_data[-1]
    eval_axes.errorbar(evaluation_points, eval_mean, eval_std, fmt="o-", capsize=7, label="Mean ± StdDev", color=color)
    eval_axes.set(xlabel="Step", ylabel="Return", title="Aggregated evaluation returns at specific steps")
    plt.ylim(min(eval_mean) - max(eval_std) - 25, max(eval_mean) + max(eval_std) + 25)
    eval_axes.legend()

    if save_fig:
        fig.savefig(f"{save_path}.png")
    plt.show()

    return


def create_gif(frames_list: List[List[np.ndarray]], save_path: str):
    """
    Creates a gif which is a number of gifs stacked horizontally.
    Args:
        frames_list: A list of episode renderings, each episode rendering itself a list of image arrays.
        save_path: path to save gif to.
    """
    gifs = [moviepy.editor.ImageSequenceClip(frames, fps=48).margin(20)for frames in frames_list]
    final_gif = moviepy.editor.clips_array([gifs])
    final_gif.write_gif(f"{save_path}.gif")

    return


# ============== END OF FILE ==============
