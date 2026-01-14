import numpy as np
from scipy.signal import find_peaks

def calculate_pos_err(sim_freq, sim_amp, exp_freq, exp_amp, height_threshold=0.1):
    """
    Calculate Peak Position Error.
    Logic: Extract peaks from both spectra and sum squared frequency differences.
    """
    # Normalize amplitudes for peak finding
    sim_amp_norm = sim_amp / np.max(np.abs(sim_amp)) if np.max(np.abs(sim_amp)) > 0 else sim_amp
    exp_amp_norm = exp_amp / np.max(np.abs(exp_amp)) if np.max(np.abs(exp_amp)) > 0 else exp_amp

    # Find peaks
    sim_peaks, _ = find_peaks(sim_amp_norm, height=height_threshold)
    exp_peaks, _ = find_peaks(exp_amp_norm, height=height_threshold)

    # Get frequencies of peaks
    sim_peak_freqs = sim_freq[sim_peaks]
    exp_peak_freqs = exp_freq[exp_peaks]

    # If number of peaks differs, we can't directly map 1-to-1 easily.
    # Simple strategy: Find nearest neighbor for each experimental peak in simulated peaks
    # Or strict requirement: Sort and take top N peaks (where N is min of both)
    
    # Strategy from todo: "Sum of squared frequency deviation between corresponding peaks"
    # We assume the peaks are sorted by frequency or intensity. 
    # Let's sort by intensity (importance) to match the most significant peaks first.
    
    # Sort peaks by amplitude (descending)
    sim_peaks_sorted_idx = np.argsort(sim_amp_norm[sim_peaks])[::-1]
    exp_peaks_sorted_idx = np.argsort(exp_amp_norm[exp_peaks])[::-1]
    
    sim_top_freqs = sim_peak_freqs[sim_peaks_sorted_idx]
    exp_top_freqs = exp_peak_freqs[exp_peaks_sorted_idx]
    
    # Match the count
    n_peaks = min(len(sim_top_freqs), len(exp_top_freqs))
    if n_peaks == 0:
        return 1e6 # Large penalty if no peaks found
        
    # Calculate error on top N peaks
    # NOTE: This assumes the "k-th strongest peak" in Sim corresponds to "k-th strongest" in Exp.
    # Alternatively, we could sort by Frequency if we know the topology matches.
    # Given "J-coupling numerical alignment", sorting by Frequency might be safer IF we are close to the solution.
    # Let's try sorting the top N peaks by frequency to ensure spatial locality.
    
    sim_top_n_freqs = np.sort(sim_top_freqs[:n_peaks])
    exp_top_n_freqs = np.sort(exp_top_freqs[:n_peaks])
    
    err = np.sum((sim_top_n_freqs - exp_top_n_freqs)**2)
    return err

def calculate_l2_err(sim_amp, exp_amp):
    """
    Calculate Spectrum L2 Norm Error.
    Logic: Normalize amplitudes and compute sum of squared residuals.
    """
    # Normalize to [0, 1] range
    def normalize(y):
        ma = np.max(y)
        mi = np.min(y)
        if ma == mi: return np.zeros_like(y)
        return (y - mi) / (ma - mi)
        
    sim_norm = normalize(sim_amp)
    exp_norm = normalize(exp_amp)
    
    return np.sum((sim_norm - exp_norm)**2)

def calculate_height_err(sim_amp, exp_amp):
    """
    Calculate Relative Height Error.
    Logic: Ratio of two strongest peaks.
    """
    # Find peaks without threshold to ensure we get something
    sim_peaks, _ = find_peaks(sim_amp, distance=1)
    exp_peaks, _ = find_peaks(exp_amp, distance=1)
    
    if len(sim_peaks) < 2 or len(exp_peaks) < 2:
        return 0.0 # Can't evaluate, return 0 or penalty? Returns 0 to ignore if only 1 peak.

    # Get amplitudes
    sim_peak_amps = sim_amp[sim_peaks]
    exp_peak_amps = exp_amp[exp_peaks]
    
    # Sort descending
    sim_sorted = np.sort(sim_peak_amps)[::-1]
    exp_sorted = np.sort(exp_peak_amps)[::-1]
    
    # Ratios
    r_sim = sim_sorted[1] / sim_sorted[0] if sim_sorted[0] != 0 else 0
    r_exp = exp_sorted[1] / exp_sorted[0] if exp_sorted[0] != 0 else 0
    
    return (r_sim - r_exp)**2

def total_cost(sim_freq, sim_amp, exp_freq, exp_amp, weights=(0.6, 0.3, 0.1)):
    """
    Calculate weighted total cost.
    """
    # 1. Alignment Check
    # Ensure frequencies are aligned. If not, interpolate sim to exp.
    # Assuming exp_freq is the "truth" grid.
    if not np.array_equal(sim_freq, exp_freq):
        sim_amp_interp = np.interp(exp_freq, sim_freq, sim_amp)
        # Use interpolated amplitude, but frequencies are now same
        common_freq = exp_freq
        sim_amp_used = sim_amp_interp
    else:
        common_freq = exp_freq
        sim_amp_used = sim_amp
        
    w1, w2, w3 = weights
    
    c1 = calculate_pos_err(common_freq, sim_amp_used, exp_freq, exp_amp)
    c2 = calculate_l2_err(sim_amp_used, exp_amp)
    c3 = calculate_height_err(sim_amp_used, exp_amp)
    
    total = w1 * c1 + w2 * c2 + w3 * c3
    return total, (c1, c2, c3)
