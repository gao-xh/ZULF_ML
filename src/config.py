import os
from dataclasses import dataclass

@dataclass
class OptimizerConfig:
    # Random Walk Steps
    step_j_coupling: float = 0.1 # Hz
    step_sg_window: int = 2      # Window size step (must be even to keep odd parity)
    step_truncation: int = 10    # Points
    
    # Cost Function Weights
    weight_pos: float = 0.6
    weight_l2: float = 0.3
    weight_height: float = 0.1
    
    # Iteration Settings
    max_iterations: int = 1000
    plot_interval: int = 50

@dataclass
class SimulationConfig:
    # Grid Settings
    max_freq: float = 400.0  # Hz
    n_points: int = 16384    # Frequency grid points
    line_width: float = 1.0  # Lorentzian broadening (Hz)

@dataclass
class PathConfig:
    # Project Root (calculated relative to this file)
    project_root: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # References
    references_dir: str = os.path.join(project_root, 'references')
    zulf_suite_path: str = os.path.join(references_dir, 'ZULF_NMR_Suite')

# Global Config Instance
OPTIMIZER_CONFIG = OptimizerConfig()
SIMULATION_CONFIG = SimulationConfig()
PATH_CONFIG = PathConfig()
