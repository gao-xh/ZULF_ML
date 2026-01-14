import numpy as np
import copy
from .simulation_wrapper import simulate_spectrum
from ..processing.signal import apply_processing, get_spectrum_from_fid
from .cost import total_cost
import matplotlib.pyplot as plt
from ..config import OPTIMIZER_CONFIG, SIMULATION_CONFIG

class ZulfOptimizer:
    def __init__(self, exp_fid, sampling_rate, spins, max_iter=1000, step_j=0.1, step_sg=2, step_trunc=10):
        """
        Args:
            exp_fid (np.ndarray): Experimental FID (Complex time domain signal).
            sampling_rate (float): Sampling rate in Hz.
            spins (list): List of spins e.g. ['1H', '13C'].
        """
        self.exp_fid = exp_fid
        self.sampling_rate = sampling_rate
        self.spins = spins
        self.max_iter = max_iter
        
        # Hyperparameters for walking
        self.step_j = step_j 
        self.step_sg = step_sg
        self.step_trunc = step_trunc
        
        self.history = []
        
    def run(self, init_j, init_sg_window, init_trunc_idx):
        """
        Run Random Walk Optimization.
        """
        
        # Current State
        current_j = np.copy(init_j)
        current_sg = int(init_sg_window)
        current_trunc = int(init_trunc_idx)
        
        # Initial Evaluate
        current_cost, _ = self.evaluate(current_j, current_sg, current_trunc)
        best_cost = current_cost
        best_params = (np.copy(current_j), current_sg, current_trunc)
        
        print(f"Initial Cost: {current_cost:.4f}")
        
        for i in range(self.max_iter):
            # 1. Propose New State
            # a. J-coupling perturbation (Symmetric)
            perturbation = np.random.normal(0, self.step_j, size=current_j.shape)
            perturbation = (perturbation + perturbation.T) / 2 
            np.fill_diagonal(perturbation, 0)
            
            new_j = current_j + perturbation
            
            # b. SG Window perturbation (Discrete Odd)
            delta_sg = np.random.choice([-self.step_sg, 0, self.step_sg])
            new_sg = current_sg + delta_sg
            if new_sg < 3: new_sg = 3 # Minimum window
            if new_sg % 2 == 0: new_sg += 1 # Ensure odd
            
            # c. Truncation perturbation (Discrete)
            delta_trunc = np.random.choice([-self.step_trunc, 0, self.step_trunc])
            new_trunc = current_trunc + delta_trunc
            if new_trunc < 10: new_trunc = 10
            if new_trunc > len(self.exp_fid): new_trunc = len(self.exp_fid)
            
            # 2. Evaluate
            try:
                new_cost, _ = self.evaluate(new_j, new_sg, new_trunc)
            except Exception as e:
                print(f"Iter {i} failed: {e}")
                new_cost = float('inf')
            
            # 3. Accept / Reject (Greedy)
            if new_cost < current_cost:
                current_j = new_j
                current_sg = new_sg
                current_trunc = new_trunc
                current_cost = new_cost
                
                # Update Best
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_params = (np.copy(current_j), current_sg, current_trunc)
                    print(f"Iter {i}: New Best Cost = {best_cost:.4f}")
            
            self.history.append(current_cost)
            
            # Optional: Visual log every 50 iters
            if i % 50 == 0:
                self.plot_progress(i, best_params)
                
        return best_params, self.history

    def evaluate(self, j_coupling, sg_window, trunc_idx):
        # 1. Process Experimental Data (Apply Truncation -> FFT -> SG Smooth)
        # Apply truncation to FID
        proc_fid = apply_processing(self.exp_fid, sg_window=None, truncation_idx=trunc_idx)
        
        # Convert to Spectrum (and apply SG smoothing here on spectrum amp)
        exp_freq, exp_amp = get_spectrum_from_fid(
            proc_fid, 
            self.sampling_rate, 
            sg_window=sg_window
        )
        
        # 2. Simulate Theoretical Spectrum (from J-coupling)
        # We need to match the frequency grid of the experiment for comparison?
        # Or interpolate. simulate_spectrum returns its own grid.
        
        # Let's verify max freq
        max_f = np.max(exp_freq)
        sim_freq, sim_amp = simulate_spectrum(
            j_coupling, 
            self.spins, 
            n_points=len(exp_freq), # Match points roughly
            max_freq=max_f,
            lw=1.0 # Line width guess, maybe this should be optimized too?
        )
        
        # 3. Calculate Cost
        loss, components = total_cost(sim_freq, sim_amp, exp_freq, exp_amp)
        return loss, components
