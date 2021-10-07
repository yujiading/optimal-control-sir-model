from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Union
from functools import partial, reduce


@dataclass
class SimulationResult:
    Is: np.ndarray
    Utility: np.ndarray = None
    dB1: np.ndarray = None
    dB2: np.ndarray = None
    Xs: np.ndarray = None
    Ss: np.ndarray = None

    @staticmethod
    def average_series(series: List[np.ndarray]) -> np.ndarray:
        sum_series = np.add.reduce(series)
        length = len(series)
        return sum_series / length

    @staticmethod
    def average_simulation_results(simulation_results: List[SimulationResult]) -> SimulationResult:
        average_Is = SimulationResult.average_series([simulation_result.Is for simulation_result in simulation_results])
        average_Utility = SimulationResult.average_series(
            [simulation_result.Utility for simulation_result in simulation_results])
        average_Xs = SimulationResult.average_series([simulation_result.Xs for simulation_result in simulation_results])
        average_simulation_result = SimulationResult(
            Is=average_Is,
            Utility=average_Utility,
            Xs=average_Xs
        )
        return average_simulation_result


@dataclass
class ModelResult:
    model_type: str
    alpha_star: np.ndarray
    average_simulation_result: SimulationResult
    all_simulation_results: List[SimulationResult] = None

    @staticmethod
    def average_model_results(results: List[ModelResult]) -> ModelResult:
        first_model_result = results[0]
        all_simulation_results = [model_result.average_simulation_result for model_result in results]
        model_result = ModelResult(
            model_type=first_model_result.model_type,
            alpha_star=None,
            average_simulation_result=SimulationResult.average_simulation_results(all_simulation_results)
        )
        return model_result
