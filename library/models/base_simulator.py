from abc import ABC, abstractmethod
from model_result import ModelResult, SimulationResult
from library.models import model_params
import math
from library.alpha_star import BaseModelAlphaStar
from typing import Type, List, Union
from functools import partial
from multiprocessing import Pool
import numpy as np
from tqdm import tqdm
from library.run_config import RunConfig


class BaseSimulator(ABC):
    """
    Run simulation for 1 infection rate & fix/changing X combination
    """

    def __init__(self, gamma, T, run_config: RunConfig):
        self.gamma = gamma
        self.T = T
        self.run_config = run_config

    @property
    @abstractmethod
    def model_type(self) -> str:
        pass

    @property
    @abstractmethod
    def alpha_star_model_class(self) -> Type[BaseModelAlphaStar]:
        pass

    @abstractmethod
    def simulate_one_trial(self, X0, I0, alpha_star) -> SimulationResult:
        pass

    @staticmethod
    def utility(I, gamma):
        return -I ** (1 - gamma) / (1 - gamma)

    def estimate_alpha(self, Xs, Ss, Is) -> np.ndarray:
        alpha_star_model = self.alpha_star_model_class(gamma=self.gamma, T=self.T)
        alpha_star = alpha_star_model.get_alpha_star(Xs=Xs, Ss=Ss, Is=Is, length=len(Is))
        return alpha_star

    def _simulate_Xs(self, ):
        # todo: add Xs
        pass

    @staticmethod
    def _simulate_dB(length):
        return np.random.normal(loc=0, scale=math.sqrt(model_params.dt), size=length)

    def run_monte_carlo_simulation(self, X0, I0, alpha_star, n_steps) -> SimulationResult:
        """
        Simulate future X and I from starting point
        """
        simulation_results = []
        if self.run_config.is_parallel:
            with Pool(self.run_config.cpu) as p:
                simulation_results = list(
                    tqdm(p.imap(partial(self.simulate_one_trial, X0=X0, I0=I0, alpha_star=alpha_star),
                                list(range(self.run_config.n_trials_monte_carlo_simulation)))))
        else:
            for i in range(self.run_config.n_trials_monte_carlo_simulation):
                simulation_result = self.simulate_one_trial(
                    X0=X0,
                    I0=I0,
                    alpha_star=alpha_star,
                )
                simulation_results.append(simulation_result)
        average_simulation_result = SimulationResult.average_simulation_results(simulation_results)
        return average_simulation_result

    def run_model_and_monte_carlo_simulation(self, Xs, Ss, Is, X0, I0) -> ModelResult:
        alpha_star = self.estimate_alpha(Xs, Ss, Is)
        n_steps = len(alpha_star)
        simulation_result = self.run_monte_carlo_simulation(
            alpha_star=alpha_star,
            X0=X0,
            I0=I0,
            n_steps=n_steps
        )
        model_result = ModelResult(
            model_type=self.model_type,
            alpha_star=alpha_star,
            simulation_result=simulation_result
        )
        return model_result
