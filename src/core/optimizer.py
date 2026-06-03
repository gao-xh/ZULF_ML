import json
from pathlib import Path

import numpy as np
import copy
from .simulation_wrapper import simulate_spectrum
from ..processing.signal import apply_processing, get_spectrum_from_fid
from .cost import total_cost
import matplotlib.pyplot as plt
from ..config import (
    OPTIMIZER_CONFIG,
    SIMULATION_CONFIG,
    optimizer_config_from_dict,
    optimizer_config_to_dict,
)

class ZulfOptimizer:
    def __init__(self, spins, sampling_rate, exp_fid=None, exp_spectrum=None, backend_name=None):
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
        self.backend_name = backend_name or SIMULATION_CONFIG.backend_name
        
        if self.exp_fid is None and self.exp_spectrum is None:
             raise ValueError("Must provide either exp_fid or exp_spectrum.")
        
        # Load Configs
        self.config = OPTIMIZER_CONFIG
        self.history = []
        self.best_params = None
        self.best_cost = float('inf')
        self.stage_history = []
        self.current_params = None
        self.completed_iterations = 0
        self.current_stage_name = None
        self.last_freq_range = None
        self.initial_params = None
        self.variable_config = None

    def set_backend(self, backend_name):
        self.backend_name = backend_name

    def get_checkpoint_payload(self):
        if self.current_params is None:
            raise ValueError("No optimizer state is available for checkpointing.")
        if self.variable_config:
            raise NotImplementedError("Checkpoint save is currently supported only for numeric-mode optimization.")

        current_j, current_sg, current_trunc, current_t2 = self.current_params
        best_j, best_sg, best_trunc, best_t2 = self.best_params if self.best_params is not None else self.current_params

        metadata = {
            'spins': list(self.spins),
            'sampling_rate': float(self.sampling_rate),
            'backend_name': self.backend_name,
            'completed_iterations': int(self.completed_iterations),
            'current_cost': float(self.history[-1]) if self.history else float(self.best_cost),
            'best_cost': float(self.best_cost),
            'current_stage_name': self.current_stage_name,
            'freq_range': None if self.last_freq_range is None else [
                None if value is None else float(value) for value in self.last_freq_range
            ],
            'current_discrete_params': [int(current_sg), int(current_trunc), float(current_t2)],
            'best_discrete_params': [int(best_sg), int(best_trunc), float(best_t2)],
            'config': optimizer_config_to_dict(self.config),
        }

        arrays = {
            'current_j': np.asarray(current_j),
            'best_j': np.asarray(best_j),
            'history': np.asarray(self.history, dtype=float),
            'stage_history': np.asarray(self.stage_history, dtype='U32'),
        }

        if self.initial_params is not None:
            arrays['initial_j'] = np.asarray(self.initial_params[0])
            metadata['initial_discrete_params'] = [
                int(self.initial_params[1]),
                int(self.initial_params[2]),
                float(self.initial_params[3]),
            ]

        if self.exp_fid is not None:
            arrays['exp_fid'] = np.asarray(self.exp_fid)
        if self.exp_spectrum is not None:
            arrays['exp_freq'] = np.asarray(self.exp_spectrum[0])
            arrays['exp_amp'] = np.asarray(self.exp_spectrum[1])

        return metadata, arrays

    def save_checkpoint(self, file_path):
        metadata, arrays = self.get_checkpoint_payload()
        target_path = Path(file_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target_path, metadata=np.array(json.dumps(metadata)), **arrays)

    @staticmethod
    def load_checkpoint(file_path):
        with np.load(file_path, allow_pickle=True) as data:
            metadata = json.loads(str(data['metadata']))
            state = {
                'spins': metadata['spins'],
                'sampling_rate': float(metadata['sampling_rate']),
                'backend_name': metadata['backend_name'],
                'completed_iterations': int(metadata['completed_iterations']),
                'current_cost': float(metadata['current_cost']),
                'best_cost': float(metadata['best_cost']),
                'current_stage_name': metadata.get('current_stage_name'),
                'freq_range': metadata.get('freq_range'),
                'config': optimizer_config_from_dict(metadata['config']),
                'current_params': (
                    np.array(data['current_j']),
                    int(metadata['current_discrete_params'][0]),
                    int(metadata['current_discrete_params'][1]),
                    float(metadata['current_discrete_params'][2]),
                ),
                'best_params': (
                    np.array(data['best_j']),
                    int(metadata['best_discrete_params'][0]),
                    int(metadata['best_discrete_params'][1]),
                    float(metadata['best_discrete_params'][2]),
                ),
                'history': data['history'].astype(float).tolist(),
                'stage_history': data['stage_history'].astype(str).tolist(),
                'exp_fid': np.array(data['exp_fid']) if 'exp_fid' in data else None,
                'exp_spectrum': (
                    np.array(data['exp_freq']),
                    np.array(data['exp_amp']),
                ) if 'exp_freq' in data and 'exp_amp' in data else None,
                'initial_params': None,
            }

            if 'initial_j' in data and 'initial_discrete_params' in metadata:
                state['initial_params'] = (
                    np.array(data['initial_j']),
                    int(metadata['initial_discrete_params'][0]),
                    int(metadata['initial_discrete_params'][1]),
                    float(metadata['initial_discrete_params'][2]),
                )

            return state

    def _build_stage_plan(self):
        configured_stages = [stage for stage in self.config.stages if stage.fraction > 0]
        if not configured_stages:
            return []

        max_iter = self.config.max_iterations
        total_fraction = sum(stage.fraction for stage in configured_stages)
        normalized = [stage.fraction / total_fraction for stage in configured_stages]
        raw_counts = [fraction * max_iter for fraction in normalized]
        counts = [int(np.floor(count)) for count in raw_counts]
        remainder = max_iter - sum(counts)

        ranked_remainders = sorted(
            enumerate(raw_counts),
            key=lambda item: item[1] - np.floor(item[1]),
            reverse=True,
        )
        for index, _ in ranked_remainders[:remainder]:
            counts[index] += 1

        stage_plan = []
        for stage, count in zip(configured_stages, counts):
            if count <= 0:
                continue
            stage_plan.append({
                'name': stage.name,
                'backend_name': stage.backend_name,
                'weights': stage.weights,
                'iterations': count,
            })
        return stage_plan

    def _get_stage_for_iteration(self, stage_plan, iteration_index):
        cumulative = 0
        for index, stage in enumerate(stage_plan):
            cumulative += stage['iterations']
            if iteration_index < cumulative:
                return index, stage, cumulative
        return len(stage_plan) - 1, stage_plan[-1], cumulative

    def _update_runtime_state(self, current_params, current_stage_name, completed_iterations, freq_range):
        self.current_params = (
            np.array(current_params[0]),
            int(current_params[1]),
            int(current_params[2]),
            float(current_params[3]),
        )
        self.current_stage_name = current_stage_name
        self.completed_iterations = int(completed_iterations)
        self.last_freq_range = freq_range

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

    def run(self, init_j, init_sg_window=None, init_trunc_idx=None, init_t2=None, callback=None, freq_range=None, variable_config=None, resume_state=None):
        """
        Run Constrained Random Walk Optimization.
        If variable_config is provided, init_j is treated as a template (object array).
        """
        self.variable_config = variable_config
        if variable_config and resume_state is not None:
            raise NotImplementedError("Checkpoint/resume currently supports numeric-mode optimization only.")

        self.last_freq_range = freq_range

        # Defaults from config if not provided
        if init_sg_window is None: init_sg_window = self.config.sg_window.initial_value
        if init_trunc_idx is None: init_trunc_idx = self.config.truncation.initial_value
        if init_t2 is None: init_t2 = self.config.t2_linewidth.initial_value
        self.initial_params = (np.copy(init_j), int(init_sg_window), int(init_trunc_idx), float(init_t2))
        
        curr_var_values = {}
        if variable_config:
            template_j = init_j
            for var, cfg in variable_config.items():
                curr_var_values[var] = cfg['initial_value']
            center_j = None
            curr_j_numeric = self._reconstruct_j(template_j, curr_var_values)
        else:
            template_j = None
            center_j = np.copy(init_j)
            curr_j_numeric = np.copy(init_j)

        curr_sg = int(init_sg_window)
        curr_trunc = int(init_trunc_idx)
        curr_t2 = float(init_t2)
        stage_plan = self._build_stage_plan()
        if not stage_plan:
            raise ValueError("Optimizer stage plan is empty.")
        start_iteration = 0

        if resume_state is not None:
            stage_plan = self._build_stage_plan()
            curr_j_numeric, curr_sg, curr_trunc, curr_t2 = resume_state['current_params']
            self.best_params = resume_state['best_params']
            self.best_cost = float(resume_state['best_cost'])
            self.history = list(resume_state.get('history', []))
            self.stage_history = list(resume_state.get('stage_history', []))
            start_iteration = int(resume_state.get('completed_iterations', 0))
            if start_iteration >= self.config.max_iterations:
                self._update_runtime_state((curr_j_numeric, curr_sg, curr_trunc, curr_t2), resume_state.get('current_stage_name'), start_iteration, freq_range)
                return self.best_params, self.history
        else:
            self.history = []
            self.stage_history = []

        current_stage_index, current_stage, current_stage_end = self._get_stage_for_iteration(stage_plan, start_iteration)
        self.backend_name = current_stage['backend_name']

        if resume_state is None:
            curr_cost, init_components, init_sim_spec, init_exp_spec = self.evaluate(
                curr_j_numeric,
                curr_sg,
                curr_trunc,
                curr_t2,
                center_j,
                freq_range,
                current_stage['backend_name'],
                current_stage['weights'],
            )

            self.best_cost = curr_cost
            self.best_params = (curr_j_numeric, curr_sg, curr_trunc, curr_t2)
            self._update_runtime_state(self.best_params, current_stage['name'], 0, freq_range)
            print(f"Initial Cost: {curr_cost:.4f}")

            if callback:
                init_vis_data = {
                    "sim_freq": init_sim_spec[0],
                    "sim_amp": init_sim_spec[1],
                    "exp_freq": init_exp_spec[0],
                    "exp_amp": init_exp_spec[1],
                    "stage": current_stage['name'],
                    "components": {
                        "c1": float(init_components[0]),
                        "c2": float(init_components[1]),
                        "c3": float(init_components[2]),
                    },
                }
                callback(-1, curr_cost, curr_cost, self.best_params, init_vis_data)
        else:
            curr_cost = float(resume_state['current_cost'])
            self._update_runtime_state((curr_j_numeric, curr_sg, curr_trunc, curr_t2), current_stage['name'], start_iteration, freq_range)

        curr_components = tuple(init_components) if resume_state is None else (np.nan, np.nan, np.nan)

        max_iter = self.config.max_iterations

        for i in range(start_iteration, max_iter):
            if i >= current_stage_end and current_stage_index + 1 < len(stage_plan):
                current_stage_index += 1
                current_stage = stage_plan[current_stage_index]
                current_stage_end += current_stage['iterations']
                self.backend_name = current_stage['backend_name']
                self.current_stage_name = current_stage['name']

            # 1. Propose New State
            new_t2 = self._perturb_continuous(curr_t2, self.config.t2_linewidth)
            new_sg = self._perturb_discrete(curr_sg, self.config.sg_window, ensure_odd=True)
            new_trunc = self._perturb_discrete(curr_trunc, self.config.truncation)
            
            new_var_values = {}
            if variable_config:
                for var, val in curr_var_values.items():
                    new_var_values[var] = self._perturb_continuous(val, variable_config[var])
                new_j_numeric = self._reconstruct_j(template_j, new_var_values)
            else:
                new_j_numeric = self._perturb_continuous(curr_j_numeric, self.config.j_coupling, is_matrix=True)

            # 2. Evaluate
            try:
                total_cost_val, new_components, new_sim_spec, new_exp_spec = self.evaluate(
                    new_j_numeric, new_sg, new_trunc, new_t2, 
                    center_j, freq_range,
                    current_stage['backend_name'],
                    current_stage['weights'],
                )

                if variable_config:
                    for var, val in new_var_values.items():
                        total_cost_val += self._calculate_penalty(val, variable_config[var])

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
                curr_j_numeric = new_j_numeric
                curr_components = tuple(new_components)
                if variable_config:
                    curr_var_values = new_var_values

                # Update Best
                if total_cost_val < self.best_cost:
                    self.best_cost = total_cost_val
                    self.best_params = (np.copy(curr_j_numeric), curr_sg, curr_trunc, curr_t2)
                    is_new_best = True
                    print(f"Iter {i}: New Best Cost = {self.best_cost:.4f}")

            self.history.append(curr_cost)
            self.stage_history.append(current_stage['name'])
            self._update_runtime_state((curr_j_numeric, curr_sg, curr_trunc, curr_t2), current_stage['name'], i + 1, freq_range)
            
            if callback:
                status_data = {
                    "stage": current_stage['name'],
                    "components": {
                        "c1": float(curr_components[0]),
                        "c2": float(curr_components[1]),
                        "c3": float(curr_components[2]),
                    },
                }
                # If new best, pass spectrum details
                vis_data = None
                if is_new_best:
                    # Current step is the best so far
                    vis_data = {
                        "sim_freq": new_sim_spec[0],
                        "sim_amp": new_sim_spec[1],
                        "exp_freq": new_exp_spec[0],
                        "exp_amp": new_exp_spec[1],
                        "stage": current_stage['name'],
                        "components": status_data['components'],
                    }
                    
                if callback(i, curr_cost, self.best_cost, self.best_params, vis_data, status_data) is False:
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
            t2_linewidth=best_t2,
            backend_name=self.backend_name,
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

    def evaluate(self, j_coupling, sg_window, trunc_idx, t2_linewidth, center_j, freq_range=None, backend_name=None, weights=None):
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
        # Fix Sweep Width: 'sweep' in Spinach is often full spectral width.
        # If exp_freq is -500 to 500, max_f is 500, sweep should be 1000?
        # Assuming ZULF spectra are low-frequency centered around zero or DC.
        # If max_f is used as sweep, it covers [-max_f/2, max_f/2]. 
        # If exp data goes up to max_f, we need sweep = 2 * max_f to cover [-max_f, max_f].
        
        max_f = np.max(np.abs(exp_freq)) if len(exp_freq) > 0 else 400.0
        # Ensure we cover the full range of experimental data
        sweep_width = 2 * max_f
        
        # Call simulate_spectrum with corrected parameter names for ZulfSimulation
        sim_freq, sim_amp = simulate_spectrum(
            j_coupling_matrix=j_coupling, 
            isotopes=self.spins, 
            npoints=len(exp_freq), # Match resolution? 
            # Note: npoints should probably be higher or matched to sweep_width / resolution
            # For now keeping it simple: match number of points of exp data (approx)
            sweep=sweep_width,
            t2_linewidth=t2_linewidth,
            backend_name=backend_name or self.backend_name,
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
            
            if np.any(mask_sim):
                eff_sim_freq = sim_freq[mask_sim]
                eff_sim_amp = sim_amp[mask_sim]
            else:
                 # Warning: Filter removed all data!
                 return float('inf'), {}, (sim_freq, sim_amp), (exp_freq, exp_amp)
            
            # Filter Experimental
            mask_exp = np.ones_like(exp_freq, dtype=bool)
            if f_min is not None: mask_exp &= (exp_freq >= f_min)
            if f_max is not None: mask_exp &= (exp_freq <= f_max)
            
            if np.any(mask_exp):
                eff_exp_freq = exp_freq[mask_exp]
                eff_exp_amp = exp_amp[mask_exp]
            else:
                 # Warning: Filter removed all exp data!
                 return float('inf'), {}, (sim_freq, sim_amp), (exp_freq, exp_amp)
        
        # 3. Calculate Fit Cost
        try:
            fit_cost, components = total_cost(
                eff_sim_freq,
                eff_sim_amp,
                eff_exp_freq,
                eff_exp_amp,
                weights=weights or (
                    self.config.weight_pos,
                    self.config.weight_l2,
                    self.config.weight_height,
                ),
                cost_config={
                    'missing_peak_penalty': self.config.cost_function.missing_peak_penalty,
                    'peak_region_weight': self.config.cost_function.peak_region_weight,
                },
            )
        except Exception as e:
            print(f"Cost calculation failed: {e}")
            fit_cost = float('inf')
            components = {}
        
        # 4. Calculate Constraint Penalty
        pen_j = self._calculate_penalty(j_coupling, self.config.j_coupling, center_val=center_j)
        pen_t2 = self._calculate_penalty(t2_linewidth, self.config.t2_linewidth)
        pen_sg = self._calculate_penalty(sg_window, self.config.sg_window)
        pen_trunc = self._calculate_penalty(trunc_idx, self.config.truncation)
        
        total = fit_cost + pen_j + pen_t2 + pen_sg + pen_trunc
        
        # Return spectra for visualization
        return total, components, (sim_freq, sim_amp), (exp_freq, exp_amp)

