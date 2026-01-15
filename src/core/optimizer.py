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
        # Config can be an object (OptimizerParamConfig) or a dict (VariableConfig)
        # Adapt access
        if isinstance(config, dict):
             step_size = config['step_size']
             min_v = config['min_value']
             max_v = config['max_value']
        else:
             step_size = config.step_size
             min_v = config.min_value
             max_v = config.max_value

        if is_matrix:
            noise = np.random.normal(0, step_size, size=value.shape)
            noise = (noise + noise.T) / 2 # Symmetrize
            if value.ndim == 2:
                np.fill_diagonal(noise, 0)
            new_val = value + noise
            # Hard Clip
            new_val = np.clip(new_val, min_v, max_v)
            return new_val
        else:
            noise = np.random.normal(0, step_size)
            new_val = value + noise
            # Hard Clip
            new_val = max(min_v, min(new_val, max_v))
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
        # Adapt Config access
        if isinstance(config, dict):
             elasticity = config['elasticity']
             min_v = config['min_value']
             max_v = config['max_value']
             default_init = config['initial_value']
        else:
             elasticity = config.elasticity
             min_v = config.min_value
             max_v = config.max_value
             default_init = config.initial_value

        if elasticity <= 0:
            return 0.0
        
        # Use config's initial_value as center if not provided specifically (e.g. for J matrix center might be init J)
        center = center_val if center_val is not None else default_init
        
        # Normalize range for magnitude independence
        # Avoid div by zero
        val_range = max_v - min_v
        if val_range == float('inf') or val_range <= 0:
            val_range = 1.0 # Fallback
            
        diff = current_val - center
        norm_diff = diff / val_range
        
        # For matrix, sum penalties
        if np.ndim(diff) > 0:
             penalty = np.sum(norm_diff**2)
        else:
             penalty = norm_diff**2
             
        return elasticity * penalty

    def _reconstruct_j(self, template, var_values):
        """Fill numerical values into template matrix based on variable dict."""
        n = template.shape[0]
        mat = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                val = template[i, j]
                if isinstance(val, (int, float, np.number)):
                    mat[i, j] = float(val)
                elif isinstance(val, str) and val in var_values:
                    mat[i, j] = var_values[val]
                else:
                    # Should handle unmapped, assume 0 or error?
                    try:
                        mat[i, j] = float(val)
                    except:
                        mat[i, j] = 0.0
        return mat

    def run(self, init_j, init_sg_window=None, init_trunc_idx=None, init_t2=None, callback=None, freq_range=None, variable_config=None):
        """
        Run Constrained Random Walk Optimization.
        If variable_config is provided, init_j is treated as a template (object array).
        """
        # Defaults from config if not provided
        if init_sg_window is None: init_sg_window = self.config.sg_window.initial_value
        if init_trunc_idx is None: init_trunc_idx = self.config.truncation.initial_value
        if init_t2 is None: init_t2 = self.config.t2_linewidth.initial_value
        
        # Variable Mode Initialization
        curr_var_values = {}
        center_var_values = {}
        if variable_config:
            # Init J is template
            template_j = init_j
            # Init Variables
            for var, cfg in variable_config.items():
                curr_var_values[var] = cfg['initial_value']
                center_var_values[var] = cfg['initial_value']
            
            # Construct first numeric J
            curr_j_numeric = self._reconstruct_j(template_j, curr_var_values)
        else:
            # Numeric Mode
            variable_config = None
            template_j = None
            center_j = np.copy(init_j) # Center for numeric matrix penalty
            curr_j_numeric = np.copy(init_j)

        # Common Parameters
        curr_sg = int(init_sg_window)
        curr_trunc = int(init_trunc_idx)
        curr_t2 = float(init_t2)
        
        # Initial Evaluate
        curr_cost, _, _, _ = self.evaluate(curr_j_numeric, curr_sg, curr_trunc, curr_t2, center_j if not variable_config else None, freq_range)
        
        # Best params format depends on mode
        # Just store the numeric matrix for uniformity in result?
        # Or store the variables too? The caller mostly cares about the J matrix.
        # Let's store (Numeric J, SG, Trunc, T2) as standardized output
        self.best_cost = curr_cost
        self.best_params = (curr_j_numeric, curr_sg, curr_trunc, curr_t2)
        
        print(f"Initial Cost: {curr_cost:.4f}")
        
        max_iter = self.config.max_iterations
        
        for i in range(max_iter):
            # 1. Propose New State
            new_t2 = self._perturb_continuous(curr_t2, self.config.t2_linewidth)
            new_sg = self._perturb_discrete(curr_sg, self.config.sg_window, ensure_odd=True)
            new_trunc = self._perturb_discrete(curr_trunc, self.config.truncation)
            
            new_j_numeric = None
            new_var_values = {}
            
            if variable_config:
                # Variable Mode: Perturb variables individually
                for var, val in curr_var_values.items():
                    cfg = variable_config[var]
                    new_var_values[var] = self._perturb_continuous(val, cfg)
                
                # Reconstruct J
                new_j_numeric = self._reconstruct_j(template_j, new_var_values)
            else:
                # Numeric Mode: Perturb whole matrix
                new_j_numeric = self._perturb_continuous(curr_j_numeric, self.config.j_coupling, is_matrix=True)

            # 2. Evaluate
            try:
                # For penalty calculation:
                # In variable mode, we penalize variables directly later.
                # In numeric mode, we pass center_j to evaluate.
                center_for_eval = None if variable_config else center_j
                
                # We calculate basic fit cost + t2/sg/trunc penalty first inside evaluate
                # But Variable Penalty needs to be added manually if in Variable Mode
                # Because evaluate() only knows about 'J Matrix penalty' via center_j argument logic which is for Numeric.
                
                # Let's calculate variable penalty separately
                total_cost_val, _, new_sim_spec, new_exp_spec = self.evaluate(
                    new_j_numeric, new_sg, new_trunc, new_t2, 
                    center_for_eval, freq_range
                )
                
                if variable_config:
                    # Add Variable Penalties
                    var_penalty = 0.0
                    for var, val in new_var_values.items():
                        var_penalty += self._calculate_penalty(val, variable_config[var])
                    
                    total_cost_val += var_penalty

            except Exception as e:
                print(f"Iter {i} failed: {e}")
                total_cost_val = float('inf')
            
            # 3. Accept / Reject (Greedy)
            is_new_best = False
            if total_cost_val < curr_cost:
                curr_sg = new_sg
                curr_trunc = new_trunc
                curr_t2 = new_t2
                curr_cost = total_cost_val
                
                if variable_config:
                    curr_var_values = new_var_values
                    # No need to update curr_j_numeric for loop (it's derived), but needed for best_params
                    curr_j_numeric = new_j_numeric 
                else:
                    curr_j_numeric = new_j_numeric

                # Update Best
                if total_cost_val < self.best_cost:
                    self.best_cost = total_cost_val
                    self.best_params = (np.copy(curr_j_numeric), curr_sg, curr_trunc, curr_t2)
                    is_new_best = True
                    print(f"Iter {i}: New Best Cost = {self.best_cost:.4f}")

            self.history.append(curr_cost)
            
            if callback:
                # If new best, pass spectrum details
                vis_data = None
                if is_new_best:
                    # Current step is the best so far
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

    def evaluate(self, j_coupling, sg_window, trunc_idx, t2_linewidth, center_j, freq_range=None):
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

        # 2a. Filter Logic for Cost Calculation
        eff_sim_freq, eff_sim_amp = sim_freq, sim_amp
        eff_exp_freq, eff_exp_amp = exp_freq, exp_amp

        if freq_range:
            f_min, f_max = freq_range
            
            # Filter Simulated
            mask_sim = np.ones_like(sim_freq, dtype=bool)
            if f_min is not None: mask_sim &= (sim_freq >= f_min)
            if f_max is not None: mask_sim &= (sim_freq <= f_max)
            eff_sim_freq = sim_freq[mask_sim]
            eff_sim_amp = sim_amp[mask_sim]
            
            # Filter Experimental
            mask_exp = np.ones_like(exp_freq, dtype=bool)
            if f_min is not None: mask_exp &= (exp_freq >= f_min)
            if f_max is not None: mask_exp &= (exp_freq <= f_max)
            eff_exp_freq = exp_freq[mask_exp]
            eff_exp_amp = exp_amp[mask_exp]
        
        # 3. Calculate Fit Cost
        fit_cost, components = total_cost(eff_sim_freq, eff_sim_amp, eff_exp_freq, eff_exp_amp)
        
        # 4. Calculate Constraint Penalty
        pen_j = self._calculate_penalty(j_coupling, self.config.j_coupling, center_val=center_j)
        pen_t2 = self._calculate_penalty(t2_linewidth, self.config.t2_linewidth)
        pen_sg = self._calculate_penalty(sg_window, self.config.sg_window)
        pen_trunc = self._calculate_penalty(trunc_idx, self.config.truncation)
        
        total = fit_cost + pen_j + pen_t2 + pen_sg + pen_trunc
        
        # Return spectra for visualization
        return total, components, (sim_freq, sim_amp), (exp_freq, exp_amp)

