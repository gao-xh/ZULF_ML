import sys
import os
import numpy as np

# Add the reference suite to path
REF_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'references', 'ZULF_NMR_Suite')
if REF_PATH not in sys.path:
    sys.path.append(REF_PATH)

try:
    from src.core.TwoD_simulation import simulation, system, interaction, parameters, pulse, environment
except ImportError as e:
    # Try absolute path import via sys.path modification in config or here
    try: 
         # This assumes ZULF_NMR_Suite is in sys.path
         from src.core.TwoD_simulation import simulation, system, interaction, parameters, pulse, environment
    except:
         print(f"Warning: ZULF Suite not found. Simulation will fail. {e}")
         simulation = None

from ..config import SIMULATION_CONFIG

def lorentzian(x, x0, gamma):
    return (1/np.pi) * (gamma / ((x - x0)**2 + gamma**2))

def simulate_spectrum(j_coupling, spins, n_points=None, max_freq=None, lw=None):
    """
    Simulate ZULF NMR Spectrum using the reference suite.
    
    Args:
        j_coupling (np.ndarray): J-coupling matrix (NxN).
        spins (list): List of isotopes e.g. ['1H', '13C'].
        n_points (int): Number of points in frequency grid.
        max_freq (float): Maximum frequency for grid (Hz).
        lw (float): Line width for broadening (Hz).
        
    Returns:
        freq_grid (np.ndarray): Frequency axis.
        spectrum (np.ndarray): Simulated amplitude.
    """
    # Use config defaults if not provided
    if n_points is None: n_points = SIMULATION_CONFIG.n_points
    if max_freq is None: max_freq = SIMULATION_CONFIG.max_freq
    if lw is None: lw = SIMULATION_CONFIG.line_width

    if simulation is None:
        raise ImportError("ZULF Suite not loaded.")
        
    # 1. Setup System
    sys_obj = system(isos=spins)
    
    # 2. Setup Interaction
    inter_obj = interaction(coupling=j_coupling)
    
    # 3. Setup Parameters
    # We need to map our inputs to what 'parameters' expects
    # parameters(npoints, zerofill, zerofill1, offset, spins, sampling_rate, ...)
    # The suite seems to do freq domain directly via eigen decomp, 
    # but 'parameters' requires some values.
    
    # Sampling rate relates to max_freq (Nyquist). Sampling = 2 * MaxFreq
    sampling_rate = 2 * max_freq
    
    # Np1 seems to be for 2D. 
    param_obj = parameters(
        npoints=n_points,
        zerofill=n_points,
        zerofill1=0,
        offset=0,
        spins=spins,
        sampling_rate=sampling_rate
    )
    
    # 4. Setup Pulse
    # Zero field usually implies sudden field drop or specific pulse.
    # We interpret 'pulse' here. 
    # For standard ZULF, maybe we don't need a pulse object if we use 'freq_domain'
    # but 'freq_domain' uses 'self.pulse' inside `self.operation(...).rho_pulse()`.
    # Let's provide a dummy or standard pulse.
    pulse_obj = pulse(
        shape="Hard", # Guessing
        duration=0,
        Bx=0, By=0, Bz=0 # Zero field?
    )
    
    # 5. Environment
    env_obj = environment(magnetic_field=0.0)
    
    # 6. Run Simulation
    sim = simulation(sys_obj, inter_obj, param_obj, pulse_obj)
    
    # freq_domain returns stick spectrum (freqs, weights)
    trans_freqs, trans_weights = sim.freq_domain(env_obj)
    
    # 7. Broaden to grid
    freq_grid = np.linspace(0, max_freq, n_points)
    spectrum = np.zeros_like(freq_grid)
    
    # Take real part of weights (or abs)
    weights = np.real(trans_weights) 
    
    # Add Lorentzians
    # Vectorized approach can be heavy if many transitions, but usually ZULF spins are few (~2-5)
    # transitions ~ (2^N)^2. For N=5, 32^2 = 1024 transitions. Fast.
    
    for f0, w in zip(trans_freqs, weights):
        if 0 <= f0 <= max_freq:
             spectrum += w * lorentzian(freq_grid, f0, lw)
             
    return freq_grid, spectrum

