import unittest
import gymnasium
from gymnasium.wrappers import FrameStack, AtariPreprocessing
from train import create_env


class TestCreateEnv(unittest.TestCase):
    def test_basic_env_creation(self):
        """Test basic environment creation without image preprocessing and video recording."""
        env_name = "CartPole-v1"
        env = create_env(env_name)
        self.assertIsInstance(env, gymnasium.Env)
        self.assertEqual(env.spec.id, env_name)

    def test_frame_stacking(self):
        """Test if the FrameStack wrapper is applied correctly."""
        env_name = "CartPole-v1"
        num_frames = 4
        env = create_env(env_name, num_frame_stacking=num_frames)

        self.assertIsInstance(env.env, FrameStack)
        obs = env.observation_space.sample()
        self.assertEqual(obs.shape, (num_frames, 4))

    def test_image_obs_preprocessing(self):
        """Test if the AtariPreprocessing wrapper is applied when image_obs is True."""
        env_name = "PongNoFrameskip-v4"
        env = create_env(env_name, image_obs=True)

        self.assertIsInstance(env.env.env, AtariPreprocessing)

    def test_record_videos(self):
        """Test if the render_mode is set to 'rgb_array' when record_video is True."""
        env_name = "CartPole-v1"
        env = create_env(env_name, record_video=True)

        self.assertEqual(env.render_mode, "rgb_array_list")


if __name__ == "__main__":
    unittest.main()
