import os
import json
import csv
import numpy as np
import struct
from pathlib import Path

def _load_nmrduino_format(folder_path):
    """
    Load data from NMRduino format (0.ini, 0.dat, ...).
    Adapted from references/Data-Process/nmr_processing_lib/nmrduino_util_fixed.py
    
    Returns:
        fid (np.ndarray): Averaged time-domain signal.
        sampling_rate (float): Sampling rate in Hz.
    """
    path = Path(folder_path)
    
    # 1. Get Sampling Rate from 0.ini
    sampling_rate = 6000.0 # fallback
    ini_file = path / '0.ini'
    if ini_file.exists():
        try:
            with open(ini_file, 'r') as f:
                found_section = False
                for line in f:
                    line = line.strip()
                    if '[NMRduino]' in line:
                         found_section = True
                    elif found_section and 'SampleRate' in line:
                         sampling_rate = float(line.split('=')[1])
                         break
        except Exception:
            pass # Use fallback

    # 2. Read Scans (.dat files)
    scan_count = 0
    summed_data = None
    
    while True:
        scan_file = path / f"{scan_count}.dat"
        if not scan_file.exists():
            break
            
        try:
            with open(scan_file, 'rb') as f:
                byte_data = bytearray(f.read())
            
            # NMRduino specific: Reverse bytes then unpack int16
            byte_data.reverse()
             # Little endian 16-bit integers
            count = len(byte_data) // 2
            int_data = struct.unpack(f'<{count}h', byte_data)
            
            # Skip header/footer (20 / 2)
            if len(int_data) > 22:
                # Note: Reference uses 20:-2
                signal = np.array(int_data[20:-2], dtype=np.float64)
                
                if summed_data is None:
                    summed_data = signal
                else:
                    # Handle length mismatch if any
                    min_len = min(len(summed_data), len(signal))
                    summed_data = summed_data[:min_len] + signal[:min_len]
        except Exception as e:
            print(f"Error reading scan {scan_count}: {e}")
            
        scan_count += 1
        
    if summed_data is None:
        raise FileNotFoundError("No valid .dat scan files found.")
        
    # Average
    avg_data = summed_data / scan_count
    
    # Flip (as per reference)
    fid = np.flip(avg_data)
    
    # Cut first point? Reference does halp[1:]
    if len(fid) > 1:
        fid = fid[1:]
        
    return fid, sampling_rate

def load_experimental_and_config(folder_path):
    """
    Load experimental data. Supports:
    1. 'Standard' ZULF Suite format: spectrum.csv (Freq, Amp) + setting.json
    2. 'NMRduino' raw format: 0.ini + 0.dat (Binary FID)
    
    Args:
        folder_path (str): Path to data folder.
        
    Returns:
        exp_data (tuple): ((freqs, amps), fid)
                          If spectrum.csv: fid is None.
                          If NMRduino: freqs, amps are computed via FFT.
        sampling_rate (float): Hz
        spins (list): Isotopes
        init_j (np.ndarray): Initial J matrix (zeros)
    """
    print(f"Loading data from: {folder_path}")
    
    # Defaults
    spins = ['1H', '13C'] 
    init_j = np.zeros((2, 2))
    
    # Check for NMRduino format first
    if os.path.exists(os.path.join(folder_path, '0.ini')) or os.path.exists(os.path.join(folder_path, '0.dat')):
        print("Detected NMRduino raw format.")
        fid, sr = _load_nmrduino_format(folder_path)
        
        # Compute Spectrum for visualization
        # Simple FFT
        spectrum_complex = np.fft.fftshift(np.fft.fft(fid))
        freqs = np.fft.fftshift(np.fft.fftfreq(len(fid), d=1/sr))
        amps = np.abs(spectrum_complex)
        
        # Subset positive/relevant freqs? ZulfOptimizer usually works with full or pos.
        # Let's return full
        
        return ((freqs, amps), fid), sr, spins, init_j

    # Fallback to Standard Spectrum CSV
    spectrum_path = os.path.join(folder_path, 'spectrum.csv')
    setting_path = os.path.join(folder_path, 'setting.json')
    
    if not os.path.exists(spectrum_path):
        raise FileNotFoundError(f"Data not found (checked spectrum.csv and NMRduino format) in {folder_path}")
        
    # 1. Load Settings
    if os.path.exists(setting_path):
        with open(setting_path, 'r', encoding='utf-8') as f:
            settings = json.load(f)
            if 'isotopes' in settings: spins = settings['isotopes']
            elif 'spins' in settings: spins = settings['spins']
         
    # 2. Load Spectrum CSV
    freqs = []
    amps = []
    
    with open(spectrum_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader, None) # Skip header if present
        # Heuristic: check if header is text. If float, it's data.
        # But for now assume header exists or handle exception
        
        # Reset file pointer if needed? No, standard usually has header.
        # Let's try reading the first line we skipped.
        if header:
             try:
                 float(header[0])
                 # It was data, oops.
                 # Re-read or just append
             except:
                 pass # It was text
             
    # Re-open safely to handle headerless
    with open(spectrum_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                try:
                    f_val = float(row[0])
                    a_val = float(row[1])
                    freqs.append(f_val)
                    amps.append(a_val)
                except ValueError:
                    continue # header or bad line
                    
    freqs = np.array(freqs)
    amps = np.array(amps)
    
    # Sort
    if len(freqs) > 0:
        sort_idx = np.argsort(freqs)
        freqs = freqs[sort_idx]
        amps = amps[sort_idx]
        max_freq = freqs.max()
        sampling_rate = 2 * max_freq
    else:
        sampling_rate = 1000.0

    # Initial J Guess
    n_spins = len(spins)
    init_j = np.zeros((n_spins, n_spins))
            
    return ((freqs, amps), None), sampling_rate, spins, init_j

def load_molecule_from_csv(path):
    """
    Load molecule structure (structure.csv) to get isotopes and J-couplings.
    Format:
        Row 0: 1H, 13C, ...
        Row 1+: J-matrix rows
    """
    print(f"Loading molecule from: {path}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Molecule file not found: {path}")
        
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        rows = list(reader)
        
    if not rows:
        raise ValueError("Empty molecule file")

    # Row 0: Isotopes
    isotopes = [iso.strip() for iso in rows[0] if iso.strip()]
    
    # Rows 1+: J Matrix
    j_matrix = []
    for row in rows[1:]:
        clean_row = [float(x) for x in row if x.strip() != '']
        if clean_row:
             j_matrix.append(clean_row)
             
    j_matrix = np.array(j_matrix)
    
    if len(isotopes) != j_matrix.shape[0]:
         print(f"Warning: Isotope count ({len(isotopes)}) != J-matrix size ({j_matrix.shape})")
         
    return isotopes, j_matrix
