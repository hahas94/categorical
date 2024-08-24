import unittest
import torch
from network import Network


class TestNetwork(unittest.TestCase):
    def setUp(self):
        """Set up the common parameters for the tests."""
        self.obs_shape_image = (1, 4, 84, 84)  # Example shape for image input
        self.obs_shape_vector = (1, 6)  # Example shape for vector input
        self.n_actions = 4
        self.n_atoms = 51
        self.n_hidden_units = 256

    def test_network_image_obs(self):
        """Test the network with image observations."""
        net = Network(self.obs_shape_image, self.n_actions, self.n_atoms, self.n_hidden_units, image_obs=True)
        sample_input = torch.zeros(*self.obs_shape_image)  # Batch size of 1
        output = net(sample_input)
        expected_shape = (1, self.n_actions, self.n_atoms)
        self.assertEqual(output.shape, expected_shape)

        sample_input2 = torch.zeros(32, 4, 84, 84)  # Batch size of 1
        output2 = net(sample_input2)
        self.assertEqual(output2.shape, (32, self.n_actions, self.n_atoms))

    def test_network_vector_obs(self):
        """Test the network with vector observations."""
        net = Network(self.obs_shape_vector, self.n_actions, self.n_atoms, self.n_hidden_units, image_obs=False)
        sample_input = torch.zeros(*self.obs_shape_vector)  # Batch size of 1
        output = net(sample_input)
        expected_shape = (1, self.n_actions, self.n_atoms)
        self.assertEqual(output.shape, expected_shape)

        sample_input2 = torch.zeros(32, 6)  # Batch size of 1
        output2 = net(sample_input2)
        expected_shape2 = (32, self.n_actions, self.n_atoms)
        self.assertEqual(output2.shape, expected_shape2)


if __name__ == '__main__':
    unittest.main()
