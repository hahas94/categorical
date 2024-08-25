"""
main.py
-------

Description:
  This file implements the main function.
"""
import os

import gymnasium
import torch

import agent
import performance
import train
import utils

# hyperparameters that can be used for several easy-to-learn envs
Easy_Envs_Params = utils.Hyperparameters(
    n_training_steps=int(5e5),
    learning_starts=int(2e4),
    train_frequency=4,
    target_update_frequency=1000,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_end=0.05,
    exploration_fraction=0.3,
    n_hidden_units=512,
    learning_rate=1e-3,
    capacity=int(2e5),
    batch_size=64,
    v_min=-100,
    v_max=100,
    n_atoms=101,
)

# hyperparameters that can be used for atari envs
Atari_Params = utils.Hyperparameters(
    n_training_steps=int(6e6),
    learning_starts=int(8e4),
    train_frequency=4,
    target_update_frequency=1000,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_end=0.05,
    exploration_fraction=0.25,
    n_hidden_units=512,
    learning_rate=2.5e-4,
    capacity=int(1e6),
    batch_size=64,
    v_min=-10,
    v_max=10,
    n_atoms=51,
)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = [6, 28, 496, 8128]  # perfect numbers
    env_name = "CartPole-v1"  # or "PongNoFrameskip-v4"
    assert env_name in gymnasium.registry.keys(), "Environment ID is incorrect."
    verboses = [True, False, False, False]  # print progress for one seed only
    record_videos = [True, False, False, False]  # record video for one seed only

    save_path = "media"  # directory for storing images and video recordings
    assert os.path.exists(os.path.join(os.getcwd(), save_path)), f"`{os.path.join(os.getcwd(), save_path)}` not exist."
    save_path = f"{save_path}/{env_name.split('-')[0]}"

    params = Easy_Envs_Params
    n_eval_episodes = 500 if env_name == "CartPole-v1" else 100
    record_statistics_fraction = 0.01
    eval_points = (0.10, 0.25, 0.50, 1.0)
    evaluation_points = [int(p * params.n_training_steps) for p in eval_points]
    n_episodes_recorded = int(1/record_statistics_fraction)
    n_frame_stack = 1
    image_obs = False  # True when training on an env with image obs
    save_fig = True

    metrics_loggers = []

    for s in range(len(seeds)):
        seed = seeds[s]
        utils.set_seed(seed)
        env = train.create_env(env_name, n_frame_stack, image_obs, n_eval_episodes, record_videos[s])
        assert isinstance(env.action_space, gymnasium.spaces.Discrete), "Only envs with discrete actions-space allowed."
        agent_s = agent.Agent(n_actions=env.action_space.n,
                              capacity=params.capacity,
                              batch_size=params.batch_size,
                              learning_rate=params.learning_rate,
                              obs_shape=env.observation_space.shape,
                              epsilon_start=params.epsilon_start,
                              epsilon_end=params.epsilon_end,
                              exploration_fraction=params.exploration_fraction,
                              training_steps=params.n_training_steps,
                              gamma=params.gamma,
                              v_min=params.v_min,
                              v_max=params.v_max,
                              n_atoms=params.n_atoms,
                              image_obs=image_obs,
                              n_hidden_units=params.n_hidden_units,
                              device=device)

        agent_metrics = train.train(seed=seed,
                                    env=env,
                                    agent=agent_s,
                                    training_steps=params.n_training_steps,
                                    train_frequency=params.train_frequency,
                                    target_update_frequency=params.target_update_frequency,
                                    learning_starts=params.learning_starts,
                                    n_eval_episodes=n_eval_episodes,
                                    evaluation_points=evaluation_points,
                                    record_statistics_fraction=record_statistics_fraction,
                                    image_obs=image_obs,
                                    device=device,
                                    verbose=verboses[s],
                                    record_video=record_videos[s],
                                    save_path=save_path)

        metrics_loggers.append(agent_metrics)

    r_env = train.create_env(env_name, n_frame_stack, image_obs, n_eval_episodes, False)
    agent_stats = performance.preprocess_results(metrics_loggers)
    random_agent_returns = train.random_agent_performance(seed=1, env=r_env, n_episodes=n_episodes_recorded)
    performance.visualize_performance(processed_data=agent_stats,
                                      training_steps=params.n_training_steps,
                                      n_episodes_recorded=n_episodes_recorded,
                                      evaluation_points=evaluation_points,
                                      baseline_returns=random_agent_returns,
                                      save_fig=save_fig,
                                      save_path=save_path)


if __name__ == '__main__':
    main()

# ============== END OF FILE ==============
