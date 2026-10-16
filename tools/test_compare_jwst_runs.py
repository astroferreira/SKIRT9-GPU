import unittest

import numpy as np

from compare_jwst_runs import metrics


class RegressionMetricsTests(unittest.TestCase):
    def test_identical_arrays(self):
        result = metrics([1., 2., 3.], [1., 2., 3.])
        self.assertEqual(result["identical_fraction"], 1)
        self.assertEqual(result["max_peak_normalized_error"], 0)
        self.assertEqual(result["relative_l2_error"], 0)

    def test_global_change_is_detected(self):
        result = metrics([1., 2.], [1.1, 2.2])
        for key in ("max_peak_normalized_error", "relative_l2_error", "relative_sum_error"):
            self.assertAlmostEqual(result[key], 0.1)

    def test_spatial_change_with_equal_totals_is_detected(self):
        result = metrics([1., 0.], [0., 1.])
        self.assertEqual(result["relative_sum_error"], 0)
        self.assertEqual(result["max_peak_normalized_error"], 1)
        self.assertGreater(result["relative_l2_error"], 1)

    def test_invalid_inputs_rejected(self):
        for a, b in (([], []), ([0.], [0.]), ([1.], [1., 2.]),
                     ([1.], [np.nan]), ([1.], [-1.])):
            with self.assertRaises(ValueError):
                metrics(a, b)


if __name__ == "__main__":
    unittest.main()
