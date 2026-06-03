import sys
import os
import numpy as np
import importlib.util
from abc import ABC, abstractmethod
from typing import Dict, Tuple, List
from scipy.linalg import expm

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


class SimulationBackend(ABC):
    name = "base"

    @abstractmethod
    def simulate_spectrum(self,
                          j_coupling_matrix: np.ndarray,
                          isotopes: List[str] = None,
                          t2_linewidth: float = 1.0,
                          field: float = 0.0,
                          sweep: float = 400.0,
                          npoints: int = 2048) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


class QuantumSimulationMixin:
    GAMMA_BY_ISOTOPE: Dict[str, float] = {
        '1H': 42.58,
        '13C': 10.71,
        '15N': 60.86,
    }

    def _validate_isotopes(self, j_coupling_matrix: np.ndarray, isotopes: List[str] = None) -> List[str]:
        n_spins = j_coupling_matrix.shape[0]
        if isotopes is None:
            isotopes = ['1H'] * n_spins
        if len(isotopes) != n_spins:
            raise ValueError(f"Number of isotopes ({len(isotopes)}) does not match matrix size ({n_spins})")
        return isotopes

    def _gamma_array(self, isotopes: List[str]) -> np.ndarray:
        gamma = []
        for isotope in isotopes:
            if isotope not in self.GAMMA_BY_ISOTOPE:
                raise ValueError(f"Unsupported isotope for python backend: {isotope}")
            gamma.append(self.GAMMA_BY_ISOTOPE[isotope])
        return np.asarray(gamma, dtype=float)

    def _single_spin_operators(self, n_spins: int):
        p_x = np.array([[0.0, 0.5], [0.5, 0.0]], dtype=complex)
        p_y = np.array([[0.0, -0.5j], [0.5j, 0.0]], dtype=complex)
        p_z = np.array([[0.5, 0.0], [0.0, -0.5]], dtype=complex)
        identity = np.eye(2, dtype=complex)

        def kron_for_spin(target_index: int, operator: np.ndarray) -> np.ndarray:
            result = np.eye(1, dtype=complex)
            for spin_index in range(n_spins):
                result = np.kron(result, operator if spin_index == target_index else identity)
            return result

        i_x = [kron_for_spin(index, p_x) for index in range(n_spins)]
        i_y = [kron_for_spin(index, p_y) for index in range(n_spins)]
        i_z = [kron_for_spin(index, p_z) for index in range(n_spins)]
        return i_x, i_y, i_z

    def _hamiltonian_terms(self, j_coupling_matrix: np.ndarray, isotopes: List[str], field: float):
        n_spins = j_coupling_matrix.shape[0]
        gamma = self._gamma_array(isotopes)
        i_x, i_y, i_z = self._single_spin_operators(n_spins)
        hamiltonian = np.zeros((2 ** n_spins, 2 ** n_spins), dtype=complex)

        for spin_index in range(n_spins):
            hamiltonian += 2 * np.pi * gamma[spin_index] * field * i_z[spin_index]

        for row in range(n_spins):
            for col in range(row + 1, n_spins):
                coupling = j_coupling_matrix[row, col]
                hamiltonian += 2 * np.pi * coupling * (
                    i_x[row] @ i_x[col] +
                    i_y[row] @ i_y[col] +
                    i_z[row] @ i_z[col]
                )

        detector = np.zeros_like(hamiltonian)
        rho0 = np.zeros_like(hamiltonian)
        for spin_index in range(n_spins):
            detector += 2 * np.pi * gamma[spin_index] * i_x[spin_index]
            rho0 += 2 * np.pi * gamma[spin_index] * i_x[spin_index]

        return hamiltonian, detector, rho0


class SpinachSimulationBackend(SimulationBackend):
    name = "spinach"

    def __init__(self):
        self.engine = None
        self.cm = None
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


class FastEigenSpectrumBackend(QuantumSimulationMixin, SimulationBackend):
    name = "fast_eigen"

    def simulate_spectrum(self,
                          j_coupling_matrix: np.ndarray,
                          isotopes: List[str] = None,
                          t2_linewidth: float = 1.0,
                          field: float = 0.0,
                          sweep: float = 400.0,
                          npoints: int = 2048) -> Tuple[np.ndarray, np.ndarray]:
        isotopes = self._validate_isotopes(j_coupling_matrix, isotopes)
        if npoints <= 1:
            raise ValueError("npoints must be greater than 1")

        hamiltonian, detector, _ = self._hamiltonian_terms(j_coupling_matrix, isotopes, field)
        eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
        detector_eig = eigenvectors.conj().T @ detector @ eigenvectors

        freq_axis = np.linspace(-sweep / 2.0, sweep / 2.0, npoints)
        amplitude = np.zeros_like(freq_axis, dtype=float)
        linewidth = max(float(t2_linewidth), 1e-3)

        for row in range(len(eigenvalues)):
            for col in range(row + 1, len(eigenvalues)):
                transition = abs(eigenvalues[col] - eigenvalues[row]) / (2 * np.pi)
                if transition > sweep / 2.0:
                    continue
                weight = abs(detector_eig[row, col]) ** 2
                if weight <= 0:
                    continue
                line = (linewidth ** 2) / ((freq_axis - transition) ** 2 + linewidth ** 2)
                amplitude += weight * line
                if transition > 0:
                    mirrored_line = (linewidth ** 2) / ((freq_axis + transition) ** 2 + linewidth ** 2)
                    amplitude += weight * mirrored_line

        return freq_axis, amplitude


class PythonFidSimulationBackend(QuantumSimulationMixin, SimulationBackend):
    name = "python_fid"

    def simulate_spectrum(self,
                          j_coupling_matrix: np.ndarray,
                          isotopes: List[str] = None,
                          t2_linewidth: float = 1.0,
                          field: float = 0.0,
                          sweep: float = 400.0,
                          npoints: int = 2048) -> Tuple[np.ndarray, np.ndarray]:
        isotopes = self._validate_isotopes(j_coupling_matrix, isotopes)
        if npoints <= 1:
            raise ValueError("npoints must be greater than 1")
        if sweep <= 0:
            raise ValueError("sweep must be positive")

        sampling_rate = sweep
        time_step = 1.0 / sampling_rate
        hamiltonian, detector, rho = self._hamiltonian_terms(j_coupling_matrix, isotopes, field)
        propagator = expm(-1j * hamiltonian * time_step)
        propagator_dagger = propagator.conj().T

        fid = np.zeros(npoints, dtype=complex)
        decay_rate = max(float(t2_linewidth), 1e-3)

        for index in range(npoints):
            time_value = index * time_step
            fid[index] = np.trace(rho @ detector) * np.exp(-time_value * decay_rate)
            rho = propagator @ rho @ propagator_dagger

        spectrum = np.fft.fftshift(np.fft.fft(fid))
        freq_axis = np.fft.fftshift(np.fft.fftfreq(npoints, d=time_step))
        amplitude = np.abs(spectrum)
        return freq_axis, amplitude


_BACKENDS = {
    SpinachSimulationBackend.name: SpinachSimulationBackend(),
    FastEigenSpectrumBackend.name: FastEigenSpectrumBackend(),
    PythonFidSimulationBackend.name: PythonFidSimulationBackend(),
}


def available_backends() -> List[str]:
    return sorted(_BACKENDS.keys())


def get_backend(name: str) -> SimulationBackend:
    if name not in _BACKENDS:
        available = ', '.join(available_backends())
        raise ValueError(f"Unknown simulation backend '{name}'. Available backends: {available}")
    return _BACKENDS[name]


def simulate_spectrum(*args, backend_name: str = 'spinach', **kwargs):
    """
    Forward calls to the requested simulation backend.
    """
    backend = get_backend(backend_name)
    return backend.simulate_spectrum(*args, **kwargs)


