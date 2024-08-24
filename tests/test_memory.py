import unittest
from memory import ReplayMemory


class TestReplayMemory(unittest.TestCase):

    def setUp(self):
        """Sets up a ReplayMemory instance before each test."""
        self.capacity = 5
        self.batch_size = 2
        self.memory = ReplayMemory(self.capacity, self.batch_size)

    def test_push_and_buffer_size(self):
        """Test that the buffer increases in size until it reaches capacity."""
        for i in range(self.capacity):
            self.memory.push(f"obs_{i}", f"action_{i}", f"reward_{i}", f"next_obs_{i}", False)
            self.assertEqual(len(self.memory), i + 1)

        # Buffer should be full now
        self.assertEqual(len(self.memory), self.capacity)

        # Test the circular nature by adding more items
        self.memory.push("obs_new", "action_new", "reward_new", "next_obs_new", True)
        self.assertEqual(self.memory._buffer[0], ("obs_new", "action_new", "reward_new", "next_obs_new", 1))

    def test_sample_raises_error(self):
        """Test that sample raises ValueError when buffer size is less than batch size."""
        self.memory.push("obs_1", "action_1", "reward_1", "next_obs_1", False)
        with self.assertRaises(ValueError):
            self.memory.sample()

    def test_sample_returns_correct_shape(self):
        """Test that the sample method returns a tuple of the correct shape."""
        # Fill the memory with enough transitions
        for i in range(self.batch_size):
            self.memory.push(f"obs_{i}", f"action_{i}", f"reward_{i}", f"next_obs_{i}", False)

        sample = self.memory.sample()
        self.assertEqual(len(sample), 5)  # Should return 5-tuple
        self.assertEqual(len(sample[0]), self.batch_size)  # Should return 2 obs


if __name__ == '__main__':
    unittest.main()
