import unittest

import numpy as np
from scipy.signal import find_peaks

from src.core.simulation_wrapper import simulate_spectrum


def strongest_peak_positions(freq_axis, amplitude, limit=3, min_height_ratio=0.15):
    max_amp = float(np.max(amplitude)) if len(amplitude) else 0.0
    if max_amp <= 0.0:
        return np.array([], dtype=float)

    peak_indices, _ = find_peaks(amplitude, height=max_amp * min_height_ratio)
    if len(peak_indices) == 0:
        return np.array([], dtype=float)

    ranked = peak_indices[np.argsort(amplitude[peak_indices])[::-1]]
    selected = np.sort(freq_axis[ranked[:limit]])
    return selected


class BackendAgreementTests(unittest.TestCase):
    def assert_peak_alignment(self, j_matrix, isotopes, sweep=400.0, npoints=257, t2_linewidth=1.0):
        fast_freq, fast_amp = simulate_spectrum(
            j_coupling_matrix=j_matrix,
            isotopes=isotopes,
            npoints=npoints,
            sweep=sweep,
            t2_linewidth=t2_linewidth,
            backend_name='fast_eigen',
        )
        fid_freq, fid_amp = simulate_spectrum(
            j_coupling_matrix=j_matrix,
            isotopes=isotopes,
            npoints=npoints,
            sweep=sweep,
            t2_linewidth=t2_linewidth,
            backend_name='python_fid',
        )

        fast_peaks = strongest_peak_positions(fast_freq, fast_amp)
        fid_peaks = strongest_peak_positions(fid_freq, fid_amp)

        self.assertGreaterEqual(len(fast_peaks), 1, 'fast_eigen should expose at least one strong peak')
        self.assertGreaterEqual(len(fid_peaks), 1, 'python_fid should expose at least one strong peak')

        common = min(len(fast_peaks), len(fid_peaks))
        peak_mismatch = np.max(np.abs(fast_peaks[:common] - fid_peaks[:common]))
        self.assertLessEqual(
            peak_mismatch,
            15.0,
            msg=f'Backend peak mismatch too large: {peak_mismatch:.3f} Hz; fast={fast_peaks}, fid={fid_peaks}',
        )

    def test_two_spin_backend_agreement(self):
        j_matrix = np.array([
            [0.0, 120.0],
            [120.0, 0.0],
        ], dtype=float)
        self.assert_peak_alignment(j_matrix, ['1H', '13C'])

    def test_three_spin_backend_agreement(self):
        j_matrix = np.array([
            [0.0, 120.0, 8.0],
            [120.0, 0.0, 5.0],
            [8.0, 5.0, 0.0],
        ], dtype=float)
        self.assert_peak_alignment(j_matrix, ['1H', '13C', '1H'], npoints=513)


if __name__ == '__main__':
    unittest.main()