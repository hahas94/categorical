import unittest
import numpy as np
import performance


class TestAggregateResults(unittest.TestCase):

    def test_basic_case(self):
        """Test aggregation with a basic list of arrays."""
        arrays_lst = [np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([7, 8, 9])]
        mean, stddev = performance.aggregate_results(arrays_lst)

        expected_mean = np.array([4, 5, 6])
        expected_stddev = np.array([2.45, 2.45, 2.45])

        np.testing.assert_array_equal(mean, expected_mean)
        np.testing.assert_array_equal(stddev, expected_stddev)


if __name__ == "__main__":
    unittest.main()
