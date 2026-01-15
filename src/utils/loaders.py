import os
import json
import csv
import numpy as np

def load_experimental_and_config(folder_path):
    """
    Load experimental data (spectrum.csv) and config (setting.json) from a folder.
    Matches the format used by the reference ZULF Suite Save/Load module.
    
    Args:
        folder_path (str): Path to the folder containing spectrum.csv and setting.json.
        
    Returns:
        exp_spectrum (tuple): (freq_axis, amp_axis) numpy arrays.
        sampling_rate (float): Sampling rate derived from settings or data.
        spins (list): List of isotopes (e.g. ['1H', '13C']) - derived or default.
        init_j (np.ndarray): Initial J-coupling guess - derived or default.
    """
    print(f"Loading data from: {folder_path}")
    
    spectrum_path = os.path.join(folder_path, 'spectrum.csv')
    setting_path = os.path.join(folder_path, 'setting.json')
    
    if not os.path.exists(spectrum_path):
        raise FileNotFoundError(f"spectrum.csv not found in {folder_path}")
    if not os.path.exists(setting_path):
        raise FileNotFoundError(f"setting.json not found in {folder_path}")
        
    # 1. Load Settings
    with open(setting_path, 'r', encoding='utf-8') as f:
        settings = json.load(f)
        
    # Defaults
    spins = ['1H', '13C'] 
    if 'isotopes' in settings:
         spins = settings['isotopes']
    elif 'spins' in settings:
         spins = settings['spins']
         
    # 2. Load Spectrum CSV
    freqs = []
    amps = []
    
    with open(spectrum_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader, None) # Skip header
        for row in reader:
            if len(row) >= 2:
                try:
                    f_val = float(row[0])
                    a_val = float(row[1])
                    freqs.append(f_val)
                    amps.append(a_val)
                except ValueError:
                    continue
                    
    freqs = np.array(freqs)
    amps = np.array(amps)
    
    # Sort just in case
    sort_idx = np.argsort(freqs)
    freqs = freqs[sort_idx]
    amps = amps[sort_idx]
    
    # Derive sampling rate if needed (2 * max_freq)
    max_freq = freqs.max() if len(freqs) > 0 else 1000.0
    sampling_rate = 2 * max_freq
    
    # Initial J Guess
    n_spins = len(spins)
    init_j = np.zeros((n_spins, n_spins))
            
    return (freqs, amps), sampling_rate, spins, init_j

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
