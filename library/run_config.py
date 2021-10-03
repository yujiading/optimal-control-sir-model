from dataclasses import dataclass, field
from typing import List
from models.model_mapper import ModelTypes
from models import model_params

@dataclass
class RunConfig:
    model: str = ModelTypes.LowOU
    gammas: List = field(default_factory=lambda: [-1, -5])
    T = 1
    X0 = model_params.X0
    I0 = model_params.eps
    S0 = model_params.S0
    is_parallel = True
    cpu: int = 8
    is_include_optimal_control: bool = True
    is_include_full_control: bool = True
    is_include_no_control: bool = True
    is_simulation: bool = True
    n_trials_simulated_data_generation: int = 1
    n_steps_simulated_data_generation: int = 100
    # n_trials_monte_carlo: int = 50000
    # seed: int = None  # Set to None to turn off seeding
    n_trials_monte_carlo_simulation: int = 50
    seed: int = 0  # Set to None to turn off seeding
