from dataclasses import dataclass, field
from typing import List
from models.model_mapper import ModelTypes

@dataclass
class RunConfig:
    model: str = ModelTypes.LowOU
    gammas: List = field(default_factory=lambda: [-1, -5])
    T = 1
    is_simulation: bool = True
    X0 = None
    I0 = None
    S0 = None
    cpu: int = 8
    n_trials_data_generation: int = 1
    n_trials: int = 10
    n_steps: int = 20
    is_include_optimal_control: bool = True
    is_include_full_control: bool = True
    is_include_no_control: bool = True
