import sys
import os

import sys
import os

# Add src directory and references to path
import sys, os
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(project_root, 'src'))
# Add ZULF Suite to path (crucial for it to find its own modules)
zulf_suite_path = os.path.join(project_root, 'references', 'ZULF_NMR_Suite')
sys.path.append(zulf_suite_path)

import numpy as np
from src.core.optimizer import ZulfOptimizer
from src.config import OPTIMIZER_CONFIG, SIMULATION_CONFIG

def generate_mock_data():
    """Generate mock experimental FID for testing."""
    # True Params
    true_j = np.array([
        [0, 140],
        [140, 0]
    ])
    spins = ['1H', '13C']
    sampling_rate = 2000 # Hz
    duration = 1.0 # sec
    
    # Simulate Spectrum directly for now as we don't have FID simulator in simulation.py yet
    # But wait, optimizer expects FID.
    # Let's mock a simple FID.
    t = np.linspace(0, duration, int(sampling_rate * duration))
    # Signal: sum of cos(2pi f t)
    # J=140 Hz (scalar coupling), in zero field usually gives lines at J, 3J/2 etc. dependent on system. 
    # For AX system (which 1H-13C is approx at ZF? No, strongly coupled).
    # Let's just create a dummy FID with some frequencies.
    frequencies = [140.0, 210.0] 
    fid = np.zeros_like(t, dtype=complex)
    for f in frequencies:
        fid += np.exp(1j * 2 * np.pi * f * t) * np.exp(-t/0.5) # Decay T2=0.5s
        
    return fid, sampling_rate, spins, true_j

def main():
    print("Starting ML_ZULF project - ZULF Optimizer...")
    
    # 1. Load Data
    print("1. Generating mock experimental data...")
    exp_fid, sampling_rate, spins, true_j = generate_mock_data()
    
    # 2. Setup Optimizer
    print("2. Setting up Optimizer...")
    optimizer = ZulfOptimizer(
        exp_fid=exp_fid,
        sampling_rate=sampling_rate,
        spins=spins
    )
    
    # 3. Random Walk
    # Initial Guess: J=100 (off by 40Hz), SG=5, Trunc=Full
    init_j = np.array([[0, 100.0], [100.0, 0]])
    init_sg = 5
    init_trunc = len(exp_fid)
    init_t2 = 1.0
    
    print("3. Starting Random Walk Optimization...")
    best_params, history = optimizer.run(init_j, init_sg, init_trunc, init_t2)
    
    best_j, best_sg, best_trunc, best_t2 = best_params
    
    print("-" * 30)
    print("Optimization Complete.")
    print(f"True J: {true_j[0,1]}")
    print(f"Best J: {best_j[0,1]}")
    print(f"Best SG: {best_sg}")
    print(f"Best Trunc: {best_trunc}")
    print(f"Best T2: {best_t2:.4f}")
    print("-" * 30)
    
    # 4. Visualization
    print("4. Generating Comparison Plot...")
    optimizer.plot_comparison(save_path="optimization_result.png")

if __name__ == "__main__":
    main()
