import unittest

import numpy as np

from estimate_jwst_packet_noise import estimate


class NoiseEstimateTests(unittest.TestCase):
    def test_corrects_noise_in_reference_mean(self):
        result = estimate(np.array([[9., 9.], [11., 11.]]), 1000, 0.1)
        self.assertAlmostEqual(result["relative_rms"], np.sqrt(4. / 198.))
        self.assertEqual(result["candidate_packets_per_phase"], 2021)

    def test_zeros_do_not_certify_convergence(self):
        result = estimate(np.zeros((2, 10)), 1000, 0.01)
        self.assertIsNone(result["candidate_packets_per_phase"])
        self.assertFalse(result["signal_power_resolved"])

    def test_identical_images_do_not_estimate_a_minimum(self):
        result = estimate(np.ones((2, 10)), 1000, 0.01)
        self.assertIsNone(result["candidate_packets_per_phase"])
        self.assertFalse(result["noise_detected"])

    def test_noise_dominated_region_requires_a_stronger_pilot(self):
        result = estimate(np.array([[0.1], [2.0]]), 1000, 0.01)
        self.assertIsNone(result["candidate_packets_per_phase"])
        self.assertTrue(result["requires_stronger_pilot"])

    def test_rejects_nonfinite_or_single_sample(self):
        for samples in (np.ones((1, 10)), np.full((2, 10), np.nan)):
            with self.assertRaises(ValueError):
                estimate(samples, 1000, 0.01)


if __name__ == "__main__":
    unittest.main()
