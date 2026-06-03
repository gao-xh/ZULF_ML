import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.signal import find_peaks

def _normalize_signal(amplitude):
    scale = float(np.max(np.abs(amplitude))) if len(amplitude) else 0.0
    if scale <= 0.0:
        return np.zeros_like(amplitude, dtype=float)
    return np.asarray(amplitude, dtype=float) / scale


def _extract_ranked_peaks(freq_axis, amplitude, height_ratio=0.1, max_peaks=12):
    amplitude = np.asarray(amplitude, dtype=float)
    magnitude = np.abs(amplitude)
    if len(magnitude) == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    max_amp = float(np.max(magnitude))
    if max_amp <= 0.0:
        return np.array([], dtype=float), np.array([], dtype=float)

    peak_indices, _ = find_peaks(
        magnitude,
        height=max_amp * height_ratio,
        prominence=max_amp * 0.03,
    )

    if len(peak_indices) == 0:
        peak_indices = np.array([int(np.argmax(magnitude))])

    ranked = peak_indices[np.argsort(magnitude[peak_indices])[::-1][:max_peaks]]
    return np.asarray(freq_axis)[ranked], magnitude[ranked] / max_amp


def calculate_pos_err(sim_freq, sim_amp, exp_freq, exp_amp, height_threshold=0.1, missing_peak_penalty=1.0):
    """
    Calculate a peak alignment error with explicit penalties for missing peaks.
    """
    sim_peak_freqs, sim_peak_heights = _extract_ranked_peaks(sim_freq, sim_amp, height_ratio=height_threshold)
    exp_peak_freqs, exp_peak_heights = _extract_ranked_peaks(exp_freq, exp_amp, height_ratio=height_threshold)

    if len(sim_peak_freqs) == 0 and len(exp_peak_freqs) == 0:
        return 0.0
    if len(sim_peak_freqs) == 0 or len(exp_peak_freqs) == 0:
        return float(max(len(sim_peak_freqs), len(exp_peak_freqs), 1) * missing_peak_penalty)

    if len(exp_freq) > 1:
        freq_step = float(np.median(np.diff(np.sort(exp_freq))))
    else:
        freq_step = 1.0
    tolerance_hz = max(2.0, 6.0 * abs(freq_step))

    distance_cost = ((exp_peak_freqs[:, None] - sim_peak_freqs[None, :]) / tolerance_hz) ** 2
    row_ind, col_ind = linear_sum_assignment(distance_cost)
    freq_err = distance_cost[row_ind, col_ind]

    height_err = np.abs(exp_peak_heights[row_ind] - sim_peak_heights[col_ind])
    matched_cost = np.mean(freq_err + 0.25 * height_err) if len(freq_err) else 0.0
    unmatched_penalty = float(abs(len(exp_peak_freqs) - len(sim_peak_freqs)) * missing_peak_penalty)
    return matched_cost + unmatched_penalty

def calculate_l2_err(sim_amp, exp_amp, peak_region_weight=3.0):
    """
    Calculate a peak-weighted shape error.
    """
    sim_norm = _normalize_signal(sim_amp)
    exp_norm = _normalize_signal(exp_amp)
    weights = 1.0 + peak_region_weight * np.abs(exp_norm)
    residual = sim_norm - exp_norm
    grad_residual = np.gradient(sim_norm) - np.gradient(exp_norm)
    return float(np.mean(weights * residual**2) + 0.5 * np.mean(weights * grad_residual**2))

def calculate_height_err(sim_amp, exp_amp):
    """
    Calculate relative peak-height mismatch for the strongest few peaks.
    """
    _, sim_peak_heights = _extract_ranked_peaks(np.arange(len(sim_amp)), sim_amp, height_ratio=0.05, max_peaks=3)
    _, exp_peak_heights = _extract_ranked_peaks(np.arange(len(exp_amp)), exp_amp, height_ratio=0.05, max_peaks=3)

    count = max(len(sim_peak_heights), len(exp_peak_heights))
    if count == 0:
        return 0.0

    sim_profile = np.zeros(count, dtype=float)
    exp_profile = np.zeros(count, dtype=float)
    sim_profile[:len(sim_peak_heights)] = sim_peak_heights
    exp_profile[:len(exp_peak_heights)] = exp_peak_heights
    return float(np.mean((sim_profile - exp_profile) ** 2))

def total_cost(sim_freq, sim_amp, exp_freq, exp_amp, weights=(0.6, 0.3, 0.1), cost_config=None):
    """
    Calculate weighted total cost.
    """
    if not np.array_equal(sim_freq, exp_freq):
        sim_amp_interp = np.interp(exp_freq, sim_freq, sim_amp)
        common_freq = exp_freq
        sim_amp_used = sim_amp_interp
    else:
        common_freq = exp_freq
        sim_amp_used = sim_amp
        
    w1, w2, w3 = weights
    cost_config = cost_config or {}
    missing_peak_penalty = cost_config.get('missing_peak_penalty', 1.0)
    peak_region_weight = cost_config.get('peak_region_weight', 3.0)
    
    c1 = calculate_pos_err(common_freq, sim_amp_used, exp_freq, exp_amp, missing_peak_penalty=missing_peak_penalty)
    c2 = calculate_l2_err(sim_amp_used, exp_amp, peak_region_weight=peak_region_weight)
    c3 = calculate_height_err(sim_amp_used, exp_amp)
    
    total = w1 * c1 + w2 * c2 + w3 * c3
    return total, (c1, c2, c3)
