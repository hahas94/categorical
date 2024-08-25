import unittest
import numpy as np
import torch
from utils import ActionInfo, to_tensor


class TestActionInfo(unittest.TestCase):
    def test_creation(self):
        """Test creating an ActionInfo instance."""
        action_info = ActionInfo(action=2, action_value=0.75, entropy=0.5)

        # Check if the fields are correctly assigned
        self.assertEqual(action_info.action, 2)
        self.assertEqual(action_info.action_value, 0.75)
        self.assertEqual(action_info.entropy, 0.5)

    def test_field_access(self):
        """Test accessing fields by name and by index."""
        action_info = ActionInfo(action=1, action_value=0.85, entropy=0.3)

        # Access by name
        self.assertEqual(action_info.action, 1)
        self.assertEqual(action_info.action_value, 0.85)
        self.assertEqual(action_info.entropy, 0.3)

        # Access by index
        self.assertEqual(action_info[0], 1)
        self.assertEqual(action_info[1], 0.85)
        self.assertEqual(action_info[2], 0.3)

    def test_immutability(self):
        """Test that ActionInfo is immutable."""
        action_info = ActionInfo(action=0, action_value=0.9, entropy=0.2)

        with self.assertRaises(AttributeError):
            action_info.action = 3

    def test_defaults(self):
        """Test that ActionInfo requires all fields to be provided."""
        with self.assertRaises(TypeError):
            ActionInfo(action=1, action_value=0.85)  # Missing entropy


class TestToTensor(unittest.TestCase):
    def test_conversion_to_tensor(self):
        """Test if the function correctly converts a numpy array to a torch tensor."""
        array = np.array([1, 2, 3, 4])
        device = torch.device('cpu')
        tensor = to_tensor(array, device)

        self.assertIsInstance(tensor, torch.Tensor)
        self.assertEqual(tensor.device, device)
        self.assertTrue(torch.equal(tensor, torch.tensor([1.0, 2.0, 3.0, 4.0])))

    def test_normalization(self):
        """Test if normalization is applied correctly."""
        array = np.array([0, 127, 255])
        device = torch.device('cpu')
        tensor = to_tensor(array, device, normalize=True)

        expected_tensor = torch.tensor([0.0, 127.0/255.0, 1.0])
        self.assertTrue(torch.allclose(tensor, expected_tensor))

    def test_new_axis(self):
        """Test if the new axis is added correctly."""
        array = np.array([1, 2, 3])
        device = torch.device('cpu')
        tensor = to_tensor(array, device, new_axis=True)

        self.assertEqual(tensor.shape, (1, 3))

    def test_nn(self):
        """Test both normalization and new-axis."""
        array = 255 * np.ones((1, 8, 8))
        device = torch.device('cpu')
        tensor = to_tensor(array, device, normalize=True, new_axis=True)

        self.assertEqual(tensor.shape, (1, 1, 8, 8))
        self.assertEqual(tensor.max(), 1.0)
        self.assertIsInstance(tensor, torch.Tensor)


if __name__ == '__main__':
    unittest.main()
