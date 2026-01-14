import numpy as np
from scipy.signal import savgol_filter

def apply_processing(time_domain_signal, sg_window, truncation_idx):
    """
    Apply signal processing: Truncation -> Smoothing (on Time Domain usually?) or Frequency domain?
    
    The spec says "SG Smoothing" and "Truncation".
    Usually Truncation happens on FID (Time Domain).
    Smoothing can happen on Frequency Domain (Spectrum) to reduce noise.
    
    Let's assume:
    1. Truncation applies to Time Domain Signal (FID).
    2. SG Filter applies to Frequency Domain Spectrum (after FFT).
    
    Wait, "Spectrum matches experimental".
    If we input time domain, we truncate, then FFT, then Smooth?
    
    Let's define standard ZULF processing flow:
    FID -> [Truncate] -> [Apodization?] -> FFT -> [Phase?] -> Magnitude/Real -> [SG Smooth] -> Spectrum
    
    The prompt mentions "SG convolution points" and "Truncation points".
    
    Args:
        time_domain_signal (np.ndarray): The raw FID.
        sg_window (int): Window length for Savitzky-Golay filter. Must be odd.
        truncation_idx (int): Index to truncate the FID.
        
    Returns:
        processed_fid (np.ndarray): Truncated FID.
    """
    
    # 1. Truncation
    # Ensure idx is within bounds
    if truncation_idx > len(time_domain_signal):
        truncation_idx = len(time_domain_signal)
    if truncation_idx < 10: # Minimum length safety
        truncation_idx = 10
        
    truncated_fid = time_domain_signal[:int(truncation_idx)]
    
    # Note: SG filter is usually for smoothing. If applied to FID, it denoises.
    # If applied to Spectrum, it smooths peaks. 
    # The doc says "SG Smoothing" in the context of "Signal Processing Parameters".
    # I will assume it applies to the final Spectrum for now, OR the user might mean Apodization?
    # But explicitly says "SG Convolution Points".
    
    # Let's return the modified FID for now. If SG is for spectrum, we need to do FFT first.
    # I'll create a full pipeline function.
    
    return truncated_fid

def get_spectrum_from_fid(fid, sampling_rate, sg_window=None, sg_order=3):
    """
    Convert FID to Spectrum via FFT, then optionally apply SG Filter.
    """
    # FFT
    spectrum = np.fft.fftshift(np.fft.fft(fid))
    freqs = np.fft.fftshift(np.fft.fftfreq(len(fid), d=1/sampling_rate))
    
    # Magnitude or Real part? ZULF usually looks at Zero-field which is often magnitude or real phased.
    # Let's use absolute magnitude for robustness unless specified.
    amp = np.abs(spectrum)
    
    # 2. SG Filter (if provided and valid)
    if sg_window is not None and sg_window > sg_order:
        if sg_window % 2 == 0:
            sg_window += 1 # Ensure odd
        try:
            amp = savgol_filter(amp, window_length=int(sg_window), polyorder=sg_order)
        except ValueError:
            pass # changes nothing if window is too small relative to order
            
    return freqs, amp
