import os
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ParameterConfig:
    """Configuration for a single optimization parameter."""
    initial_value: float = 0.0       # Center/Start value
    min_value: float = -float('inf') # Hard Lower Bound
    max_value: float = float('inf')  # Hard Upper Bound
    step_size: float = 1.0           # Std for Gaussian or Step for Discrete
    elasticity: float = 0.0          # Soft Constraint Weight (Penalty)

@dataclass
class OptimizerConfig:
    # Parameter Configurations
    # J-Coupling (Continuous)
    j_coupling: ParameterConfig = field(default_factory=lambda: ParameterConfig(
        step_size=0.1, min_value=0.0, max_value=300.0, elasticity=0.1
    ))
    
    # T2 Linewidth (Continuous) - OLD default was fixed 1.0
    t2_linewidth: ParameterConfig = field(default_factory=lambda: ParameterConfig(
        initial_value=1.0, min_value=0.1, max_value=10.0, step_size=0.1, elasticity=0.5
    ))

    # SG Window (Discrete)
    sg_window: ParameterConfig = field(default_factory=lambda: ParameterConfig(
        initial_value=5, min_value=3, max_value=21, step_size=2, elasticity=0.1
    ))

    # Truncation (Discrete)
    truncation: ParameterConfig = field(default_factory=lambda: ParameterConfig(
        initial_value=1000, min_value=10, max_value=16384, step_size=10, elasticity=0.0
    ))

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
    # line_width moved to OptimizerConfig as a variable

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
