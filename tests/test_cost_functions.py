import unittest

import numpy as np

from src.core.cost import calculate_pos_err, total_cost


class CostFunctionTests(unittest.TestCase):
    def setUp(self):
        self.freq = np.linspace(0.0, 100.0, 1001)
        self.exp_amp = (
            np.exp(-0.5 * ((self.freq - 20.0) / 1.5) ** 2)
            + 0.8 * np.exp(-0.5 * ((self.freq - 50.0) / 2.0) ** 2)
            + 0.5 * np.exp(-0.5 * ((self.freq - 80.0) / 1.5) ** 2)
        )

    def test_total_cost_is_near_zero_for_identical_spectra(self):
        total, components = total_cost(self.freq, self.exp_amp, self.freq, self.exp_amp)
        self.assertLess(total, 1e-8)
        self.assertTrue(all(component < 1e-8 for component in components))

    def test_missing_peak_cost_exceeds_small_shift_cost(self):
        shifted_amp = (
            np.exp(-0.5 * ((self.freq - 21.5) / 1.5) ** 2)
            + 0.8 * np.exp(-0.5 * ((self.freq - 51.0) / 2.0) ** 2)
            + 0.5 * np.exp(-0.5 * ((self.freq - 79.5) / 1.5) ** 2)
        )
        missing_peak_amp = (
            np.exp(-0.5 * ((self.freq - 20.0) / 1.5) ** 2)
            + 0.8 * np.exp(-0.5 * ((self.freq - 50.0) / 2.0) ** 2)
        )

        shifted_cost = calculate_pos_err(self.freq, shifted_amp, self.freq, self.exp_amp)
        missing_cost = calculate_pos_err(self.freq, missing_peak_amp, self.freq, self.exp_amp)

        self.assertGreater(missing_cost, shifted_cost)

    def test_total_cost_penalizes_wrong_peak_structure(self):
        wrong_amp = np.exp(-0.5 * ((self.freq - 35.0) / 6.0) ** 2)
        correct_total, _ = total_cost(self.freq, self.exp_amp, self.freq, self.exp_amp)
        wrong_total, _ = total_cost(self.freq, wrong_amp, self.freq, self.exp_amp)

        self.assertGreater(wrong_total, correct_total + 0.5)


if __name__ == '__main__':
    unittest.main()