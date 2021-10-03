from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Union


@dataclass
class SimulationResult:
    I: pd.Series
    Utility: pd.Series
    db2: pd.Series = None
    X: pd.Series = None

    @staticmethod
    def average_series(series: pd.Series) -> pd.Series:
        pass

    @staticmethod
    def average_simulation_results(results: List[SimulationResult]) -> SimulationResult:
        pass


@dataclass
class ModelResult:
    model_type: str
    alpha_star: Union[pd.Series, float]
    simulation_result: SimulationResult

    @staticmethod
    def average_model_results(results: List[ModelResult]) -> ModelResult:
        pass