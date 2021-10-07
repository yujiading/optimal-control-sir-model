from dataclasses import dataclass, field
from typing import List
from library.models.model_mapper import ModelTypes
from library.models import model_params


@dataclass
class RunConfig:
    model: str = ModelTypes.ModerateConst
    gammas: List = field(default_factory=lambda: [-0.5,-1,-2,-3,-4,-5])
    # gammas: List = field(default_factory=lambda: [-1, -2])
    T = 1
    # X0 = model_params.X0
    X0 = -1
    I0 = model_params.eps
    S0 = 0.7
    alpha_fix = 0.25
    is_parallel = True
    cpu: int = 8
    is_include_optimal_control: bool = True
    is_include_full_control: bool = True
    is_include_no_control: bool = True
    is_simulation: bool = True
    n_steps_simulated_data_generation: int = 100
    n_trials_simulated_data_generation: int = 100   #100
    # seed: int = None  # Set to None to turn off seeding
    n_trials_monte_carlo_simulation: int = 10000  #10000
    # seed: int = 0  # Set to None to turn off seeding
