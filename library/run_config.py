from dataclasses import dataclass, field
from typing import List
from library.models.model_mapper import ModelTypes
from library.models import model_params


@dataclass
class RunConfig:
    model: str = ModelTypes.ModerateConst
    gammas: List = field(default_factory=lambda: [-0.5,-1,-2,-3,-4,-5])
    T = 1
    # X0 = model_params.X0
    X0 = -1
    # I0 = model_params.eps
    I0:float = 0.02   # low infection regime 0.01; moderate infection regime 0.02
    S0 = 0.7
    alpha_fix = 0.25
    is_parallel:bool = False
    cpu: int = 8
    is_include_optimal_control: bool = True
    is_include_full_control: bool = True
    is_include_no_control: bool = True
    is_simulation: bool = False # real data use False
    n_steps_simulated_data_generation: int = 100
    n_trials_simulated_data_generation: int = 1  # 100 for the simulation plots; 1 for real data one scenario
    n_trials_monte_carlo_simulation: int = 1  # 10000 for the simulation plots; 1 for real data one scenario
