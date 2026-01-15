import numpy as np
import copy
from .simulation_wrapper import simulate_spectrum
from ..processing.signal import apply_processing, get_spectrum_from_fid
from .cost import total_cost
import matplotlib.pyplot as plt
from ..config import OPTIMIZER_CONFIG, SIMULATION_CONFIG

class ZulfOptimizer:
    def __init__(self, spins, sampling_rate, exp_fid=None, exp_spectrum=None):
        """
        Args:
            spins (list): List of spins e.g. ['1H', '13C'].
            sampling_rate (float): Sampling rate in Hz.
            exp_fid (np.ndarray, optional): Experimental FID (Complex time domain signal).
            exp_spectrum (tuple, optional): (freq_axis, amp_axis) if FID is not available.
        """
        self.exp_fid = exp_fid
        self.exp_spectrum = exp_spectrum
        self.sampling_rate = sampling_rate
        self.spins = spins
        
        if self.exp_fid is None and self.exp_spectrum is None:
             raise ValueError("Must provide either exp_fid or exp_spectrum.")
        
        # Load Configs
        self.config = OPTIMIZER_CONFIG
        self.history = []
        self.best_params = None
        self.best_cost = float('inf')

    def _perturb_continuous(self, value, config, is_matrix=False):
        """Perturb continuous variable with Gaussian noise + Constraints."""
        if is_matrix:
            noise = np.random.normal(0, config.step_size, size=value.shape)
            noise = (noise + noise.T) / 2 # Symmetrize
            if value.ndim == 2:
                np.fill_diagonal(noise, 0)
            new_val = value + noise
            # Hard Clip
            new_val = np.clip(new_val, config.min_value, config.max_value)
            return new_val
        else:
            noise = np.random.normal(0, config.step_size)
            new_val = value + noise
            # Hard Clip
            new_val = max(config.min_value, min(new_val, config.max_value))
            return new_val

    def _perturb_discrete(self, value, config, ensure_odd=False):
        """Perturb discrete variable with Step + Constraints."""
        step = int(config.step_size)
        delta = np.random.choice([-step, 0, step])
        new_val = int(value + delta)
        
        # Constraints
        new_val = max(int(config.min_value), min(new_val, int(config.max_value)))
        
        if ensure_odd and new_val % 2 == 0:
            if new_val + 1 <= config.max_value:
                new_val += 1
            else:
                new_val -= 1
        
        return new_val

    def _calculate_penalty(self, current_val, config, center_val=None):
        """Calculate soft elasticity penalty: weight * ((val - center)/range)^2."""
        if config.elasticity <= 0:
            return 0.0
        
        # Use config's initial_value as center if not provided specifically (e.g. for J matrix center might be init J)
        center = center_val if center_val is not None else config.initial_value
        
        # Normalize range for magnitude independence
        # Avoid div by zero
        val_range = config.max_value - config.min_value
        if val_range == float('inf') or val_range <= 0:
            val_range = 1.0 # Fallback
            
        diff = current_val - center
        norm_diff = diff / val_range
        
        # For matrix, sum penalties
        if np.ndim(diff) > 0:
             penalty = np.sum(norm_diff**2)
        else:
             penalty = norm_diff**2
             
        return config.elasticity * penalty

    def run(self, init_j, init_sg_window=None, init_trunc_idx=None, init_t2=None, callback=None):
        """
        Run Constrained Random Walk Optimization.
        """
        # Defaults from config if not provided
        if init_sg_window is None: init_sg_window = self.config.sg_window.initial_value
        if init_trunc_idx is None: init_trunc_idx = self.config.truncation.initial_value
        if init_t2 is None: init_t2 = self.config.t2_linewidth.initial_value
        
        # Store Centers for Elasticity (J is specific)
        center_j = np.copy(init_j)
        
        # Current State
        curr_j = np.copy(init_j)
        curr_sg = int(init_sg_window)
        curr_trunc = int(init_trunc_idx)
        curr_t2 = float(init_t2)
        
        # Initial Evaluate
        curr_cost, _, _, _ = self.evaluate(curr_j, curr_sg, curr_trunc, curr_t2, center_j)
        
        self.best_cost = curr_cost
        self.best_params = (np.copy(curr_j), curr_sg, curr_trunc, curr_t2)
        
        print(f"Initial Cost: {curr_cost:.4f}")
        
        max_iter = self.config.max_iterations
        
        for i in range(max_iter):
            # 1. Propose New State
            new_j = self._perturb_continuous(curr_j, self.config.j_coupling, is_matrix=True)
            new_t2 = self._perturb_continuous(curr_t2, self.config.t2_linewidth)
            new_sg = self._perturb_discrete(curr_sg, self.config.sg_window, ensure_odd=True)
            new_trunc = self._perturb_discrete(curr_trunc, self.config.truncation)
            
            # 2. Evaluate
            try:
                new_cost, _, new_sim_spec, new_exp_spec = self.evaluate(new_j, new_sg, new_trunc, new_t2, center_j)
            except Exception as e:
                print(f"Iter {i} failed: {e}")
                new_cost = float('inf')
            
            # 3. Accept / Reject (Greedy)
            if new_cost < curr_cost:
                curr_j = new_j
                curr_sg = new_sg
                curr_trunc = new_trunc
                curr_t2 = new_t2
                curr_cost = new_cost
                
                # Update Best
                if new_cost < self.best_cost:
                    self.best_cost = new_cost
                    self.best_params = (np.copy(curr_j), curr_sg, curr_trunc, curr_t2)
                    print(f"Iter {i}: New Best Cost = {self.best_cost:.4f}")

            self.history.append(curr_cost)
            
            if callback:
                # If new best, pass spectrum details
                vis_data = None
                if curr_cost == self.best_cost:
                    # Current step is the best so far (or equal)
                    vis_data = {
                        "sim_freq": new_sim_spec[0],
                        "sim_amp": new_sim_spec[1],
                        "exp_freq": new_exp_spec[0],
                        "exp_amp": new_exp_spec[1]
                    }
                    
                if callback(i, curr_cost, self.best_cost, self.best_params, vis_data) is False:
                    print("Optimization stopped by callback.")
                    break

            if i % self.config.plot_interval == 0:
                pass # self.plot_progress(i)
                
        return self.best_params, self.history

    def plot_comparison(self, save_path=None):
        """
        Plot the comparison between Experimental and Simulated spectra
        using the best parameters found so far.
        """
        if self.best_params is None:
            print("No optimization result to plot.")
            return

        best_j, best_sg, best_trunc, best_t2 = self.best_params
        center_j = None  # Not needed for re-evaluation, only for penalty

        # Re-generate Experimental Spectrum
        if self.exp_fid is not None:
            proc_fid = apply_processing(self.exp_fid, sg_window=None, truncation_idx=best_trunc)
            exp_freq, exp_amp = get_spectrum_from_fid(
                proc_fid, 
                self.sampling_rate, 
                sg_window=best_sg
            )
        else:
            exp_freq, exp_amp = self.exp_spectrum

        # Re-generate Simulated Spectrum
        max_f = np.max(exp_freq) if len(exp_freq) > 0 else 400.0
        # Call simulate_spectrum with corrected parameter names
        sim_freq, sim_amp = simulate_spectrum(
            j_coupling_matrix=best_j, 
            isotopes=self.spins, 
            npoints=len(exp_freq),
            sweep=max_f,
            t2_linewidth=best_t2 
        )
        
        # Plotting
        plt.figure(figsize=(10, 6))
        
        # Experimental
        # Normalize for visualization if needed, or keeping scaling
        plt.plot(exp_freq, exp_amp, label='Experimental (Smoothed)', alpha=0.7)
        
        # Simulated
        # Since simulation might have different amplitude scale, we might want to scale it to match
        # Naive scaling: fit sim to exp max
        if np.max(sim_amp) > 0 and np.max(exp_amp) > 0:
             scale_factor = np.max(exp_amp) / np.max(sim_amp)
             sim_amp_scaled = sim_amp * scale_factor
             plt.plot(sim_freq, sim_amp_scaled, label='Simulated (Best Fit)', linestyle='--')
        else:
             plt.plot(sim_freq, sim_amp, label='Simulated (Best Fit)', linestyle='--')
             
        plt.title(f"Optimization Result\nCost: {self.best_cost:.4f} | T2: {best_t2:.2f}Hz | SG: {best_sg}")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(0, max_f) # ZULF is low frequency usually
        
        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
        else:
            plt.show()

    def evaluate(self, j_coupling, sg_window, trunc_idx, t2_linewidth, center_j):
        # 1. Process Experimental Data
        if self.exp_fid is not None:
             proc_fid = apply_processing(self.exp_fid, sg_window=None, truncation_idx=trunc_idx)
             
             exp_freq, exp_amp = get_spectrum_from_fid(
                 proc_fid, 
                 self.sampling_rate, 
                 sg_window=sg_window
             )
        else:
             # Use pre-loaded spectrum directly
             exp_freq, exp_amp = self.exp_spectrum
        
        # 2. Simulate Theoretical Spectrum (use optimized T2)
        max_f = np.max(exp_freq) if len(exp_freq) > 0 else 400.0
        
        # Call simulate_spectrum with corrected parameter names for ZulfSimulation
        # Mapping: max_freq -> sweep, n_points -> npoints, lw -> t2_linewidth
        sim_freq, sim_amp = simulate_spectrum(
            j_coupling_matrix=j_coupling, 
            isotopes=self.spins, 
            npoints=len(exp_freq),
            sweep=max_f,
            t2_linewidth=t2_linewidth 
        )
        
        # 3. Calculate Fit Cost
        fit_cost, components = total_cost(sim_freq, sim_amp, exp_freq, exp_amp)
        
        # 4. Calculate Constraint Penalty
        pen_j = self._calculate_penalty(j_coupling, self.config.j_coupling, center_val=center_j)
        pen_t2 = self._calculate_penalty(t2_linewidth, self.config.t2_linewidth)
        pen_sg = self._calculate_penalty(sg_window, self.config.sg_window)
        pen_trunc = self._calculate_penalty(trunc_idx, self.config.truncation)
        
        total = fit_cost + pen_j + pen_t2 + pen_sg + pen_trunc
        
        # Return spectra for visualization
        return total, components, (sim_freq, sim_amp), (exp_freq, exp_amp)

