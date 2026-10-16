"""Focused checks for the spectral display convolution, independent of a simulation."""

import unittest

import numpy as np

from render_jwst_observation import smooth_spectrum


class SpectralConvolutionTests(unittest.TestCase):
    def setUp(self):
        self.wavelength = np.exp(np.linspace(np.log(1.9), np.log(2.1), 10001))

    def test_preserves_constant_flux_per_log_interval(self):
        flux = np.full_like(self.wavelength, 2.5)
        np.testing.assert_allclose(smooth_spectrum(self.wavelength, flux, 1000), flux)

    def test_line_width_and_integrated_flux(self):
        flux = np.zeros_like(self.wavelength)
        center = len(flux) // 2
        flux[center] = 1
        result = smooth_spectrum(self.wavelength, flux, 1000)
        self.assertAlmostEqual(float(result.sum()), 1, places=12)
        above_half = np.flatnonzero(result >= result.max() / 2)
        width = self.wavelength[above_half[-1]] - self.wavelength[above_half[0]]
        measured_resolution = self.wavelength[center] / width
        self.assertLess(abs(measured_resolution / 1000 - 1), 0.03)

    def test_rejects_invalid_grids(self):
        for wave in (self.wavelength[::-1], np.array([1., 2., 3.]), np.array([0., 1., 2.])):
            with self.assertRaises(ValueError):
                smooth_spectrum(wave, np.ones_like(wave), 1000)

    def test_rejects_invalid_flux_and_resolution(self):
        with self.assertRaises(ValueError):
            smooth_spectrum(self.wavelength, np.full_like(self.wavelength, np.nan), 1000)
        with self.assertRaises(ValueError):
            smooth_spectrum(self.wavelength, np.ones_like(self.wavelength), 0)


if __name__ == "__main__":
    unittest.main()
