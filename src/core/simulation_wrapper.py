import sys
import os
import numpy as np
import importlib.util
from typing import Tuple, List, Optional

# --- Dynamic Import of spinach_bridge from References ---
# This allows us to use the reference implementation without copying it
# and avoids namespace conflicts with our own 'src' package.

current_dir = os.path.dirname(os.path.abspath(__file__))
# current: .../ML_ZULF/src/core
project_root = os.path.dirname(os.path.dirname(current_dir))
# project: .../ML_ZULF
bridge_path = os.path.join(project_root, "references", "ZULF_NMR_Suite", "src", "core", "spinach_bridge.py")

spinach_bridge = None
try:
    if os.path.exists(bridge_path):
        spec = importlib.util.spec_from_file_location("spinach_bridge", bridge_path)
        if spec and spec.loader:
            spinach_bridge = importlib.util.module_from_spec(spec)
            sys.modules["spinach_bridge"] = spinach_bridge
            spec.loader.exec_module(spinach_bridge)
    else:
        print(f"Warning: spinach_bridge.py not found at {bridge_path}")
except ImportError as e:
    print(f"Warning: explicit import of spinach_bridge failed: {e}")
except Exception as e:
    print(f"Warning: Unexpected error loading spinach_bridge: {e}")


class ZulfSimulation:
    def __init__(self):
        self.engine = None
        if spinach_bridge is None:
            print("Error: spinach_bridge module is not loaded. Simulation will fail.")

    def start_engine(self):
        """Initializes the MATLAB engine via spinach_bridge."""
        if spinach_bridge is None:
            raise RuntimeError("Cannot start engine: spinach_bridge not loaded.")
        
        if self.engine is not None:
            return

        try:
            print("Starting MATLAB engine...")
            self.cm = spinach_bridge.spinach_eng(clean=True) 
            self.engine = self.cm.__enter__()
            spinach_bridge.call_spinach.default_eng = self.engine
            print("MATLAB engine started.")
        except Exception as e:
            raise RuntimeError(f"Failed to start MATLAB engine: {e}")

    def stop_engine(self):
        if self.engine:
            try:
                self.cm.__exit__(None, None, None)
            except:
                pass
            self.engine = None

    def simulate_spectrum(self, 
                          j_coupling_matrix: np.ndarray, 
                          isotopes: List[str] = None,
                          t2_linewidth: float = 1.0,
                          field: float = 0.0,
                          sweep: float = 400.0,
                          npoints: int = 2048) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulates the ZULF NMR spectrum using the Spinach bridge.
        """
        if spinach_bridge is None or (self.engine is None and not self._try_start()):
             # Fallback mock for testing without MATLAB
            # print("Warning: Utilizing fallback simulation (random data) due to missing bridge/engine.")
            # raise RuntimeError("Matlab Engine not available")
            pass

        if self.engine is None:
             self.start_engine()

        # Input validation
        n_spins = j_coupling_matrix.shape[0]
        if isotopes is None:
            isotopes = ['1H'] * n_spins
        
        if len(isotopes) != n_spins:
            raise ValueError(f"Number of isotopes ({len(isotopes)}) does not match matrix size ({n_spins})")

        # Shortcuts
        SYS = spinach_bridge.sys
        BAS = spinach_bridge.bas
        INTER = spinach_bridge.inter
        PAR = spinach_bridge.parameters
        SIM = spinach_bridge.sim
        DATA = spinach_bridge.data

        var_prefix = "opt_" 

        try:
            # 1. System Setup
            sys_obj = SYS(self.engine, var_prefix=var_prefix)
            sys_obj.isotopes(isotopes)
            sys_obj.magnet(field)

            # 2. Basis Setup
            bas_obj = BAS(self.engine, var_prefix=var_prefix)
            bas_obj.formalism('zeeman-hilb')
            bas_obj.approximation('none')

            # 3. Interactions
            inter_obj = INTER(self.engine, var_prefix=var_prefix)
            inter_obj.coupling_array(j_coupling_matrix, validate=False, use_gpu=False)

            # 4. Parameters
            par_obj = PAR(self.engine, var_prefix=var_prefix)
            par_obj.sweep(sweep)
            par_obj.npoints(npoints)
            par_obj.zerofill(8192) 
            par_obj.offset(0)
            par_obj.spins([isotopes[0]]) 
            par_obj.axis_units('Hz')
            par_obj.invert_axis(0)
            par_obj.flip_angle(np.pi/2)
            par_obj.detection('uniaxial')

            # 5. Run Simulation
            sim_obj = SIM(self.engine, var_prefix=var_prefix)
            sim_obj.create()
            sim_obj.liquid('zerofield', 'labframe')

            # 6. Process Data
            data_obj = DATA(self.engine, var_prefix=var_prefix)
            data_obj.apodisation([('exp', t2_linewidth)], use_gpu=False)
            
            spectrum = data_obj.spectrum(use_gpu=False)
            freq_axis = data_obj.freq(spectrum)

            return np.array(freq_axis).flatten(), np.real(np.array(spectrum)).flatten()

        except Exception as e:
            print(f"Simulation failed: {e}")
            raise

    def _try_start(self):
        try:
            self.start_engine()
            return True
        except:
            return False

# Global instance for easier import
zulf_sim = ZulfSimulation()

def simulate_spectrum(*args, **kwargs):
    """
    Wrapper function to forward calls to the ZulfSimulation instance.
    Usage: simulate_spectrum(j_coupling_matrix=..., isotopes=..., ...)
    """
    return zulf_sim.simulate_spectrum(*args, **kwargs)


