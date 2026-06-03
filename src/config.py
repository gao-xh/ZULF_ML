import os
from dataclasses import asdict, dataclass, field
from typing import List, Optional, Tuple

@dataclass
class ParameterConfig:
    """Configuration for a single optimization parameter."""
    initial_value: float = 0.0       # Center/Start value
    min_value: float = -float('inf') # Hard Lower Bound
    max_value: float = float('inf')  # Hard Upper Bound
    step_size: float = 1.0           # Std for Gaussian or Step for Discrete
    elasticity: float = 0.0          # Soft Constraint Weight (Penalty)


@dataclass
class OptimizationStageConfig:
    name: str
    backend_name: str
    fraction: float
    weights: Tuple[float, float, float]


@dataclass
class CostFunctionConfig:
    missing_peak_penalty: float = 1.0
    peak_region_weight: float = 3.0

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
    cost_function: CostFunctionConfig = field(default_factory=CostFunctionConfig)
    
    # Iteration Settings
    max_iterations: int = 1000
    plot_interval: int = 50
    stages: List[OptimizationStageConfig] = field(default_factory=lambda: [
        OptimizationStageConfig(
            name='stage_a',
            backend_name='fast_eigen',
            fraction=0.4,
            weights=(0.85, 0.15, 0.0),
        ),
        OptimizationStageConfig(
            name='stage_b',
            backend_name='python_fid',
            fraction=0.6,
            weights=(0.2, 0.65, 0.15),
        ),
    ])

@dataclass
class SimulationConfig:
    # Grid Settings
    max_freq: float = 400.0  # Hz
    n_points: int = 16384    # Frequency grid points
    backend_name: str = 'spinach'
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


def optimizer_config_to_dict(config: OptimizerConfig):
    return asdict(config)


def optimizer_config_from_dict(data):
    config = OptimizerConfig()

    if 'j_coupling' in data:
        config.j_coupling = ParameterConfig(**data['j_coupling'])
    if 't2_linewidth' in data:
        config.t2_linewidth = ParameterConfig(**data['t2_linewidth'])
    if 'sg_window' in data:
        config.sg_window = ParameterConfig(**data['sg_window'])
    if 'truncation' in data:
        config.truncation = ParameterConfig(**data['truncation'])
    if 'cost_function' in data:
        config.cost_function = CostFunctionConfig(**data['cost_function'])

    config.weight_pos = data.get('weight_pos', config.weight_pos)
    config.weight_l2 = data.get('weight_l2', config.weight_l2)
    config.weight_height = data.get('weight_height', config.weight_height)
    config.max_iterations = data.get('max_iterations', config.max_iterations)
    config.plot_interval = data.get('plot_interval', config.plot_interval)

    if 'stages' in data:
        config.stages = [OptimizationStageConfig(**stage) for stage in data['stages']]

    return config
