import math
from abc import ABC, abstractmethod
from multiprocessing import Pool, Array
from typing import Type, List, Tuple
from functools import partial
import ctypes
import numpy as np
from tqdm import tqdm

from library.alpha_star import BaseModelAlphaStar
from library.models import model_params
from library.run_config import RunConfig
from library.models.model_result import ModelResult, SimulationResult
from library.I_functions import IFunctions


class BaseSimulator(ABC):
    """
    Run simulation for 1 combination of infection regime & fix/changing X
    """

    def __init__(self, gamma, run_config: RunConfig):
        self.gamma = gamma
        self.run_config = run_config
        self.X0 = self.run_config.X0
        self.S0 = self.run_config.S0

    @property
    @abstractmethod
    def model_type(self) -> str:
        pass

    @property
    @abstractmethod
    def alpha_star_model_class(self) -> Type[BaseModelAlphaStar]:
        pass

    @staticmethod
    def _simulate_dB(length):
        return np.random.normal(loc=0, scale=math.sqrt(model_params.dt), size=length)

    @staticmethod
    def next_X(last_X, last_dB2):
        ret = last_X + model_params.lambda_x * (
                model_params.X_bar - last_X) * model_params.dt - model_params.sigma_x * last_dB2
        return ret

    @staticmethod
    @abstractmethod
    def next_S(last_S, last_I, last_dB1):
        pass

    @staticmethod
    @abstractmethod
    def next_I(last_alpha, last_I, last_S, last_X, last_dB1, last_dB2):
        pass

    def estimate_alpha(self, Xs, Ss, Is) -> np.ndarray:
        alpha_star_model = self.alpha_star_model_class(
            gamma=self.gamma,
            run_config=self.run_config,
            T=self.run_config.T
        )
        alpha_star = alpha_star_model.get_alpha_star(Xs=Xs, Ss=Ss, Is=Is, length=len(Is))
        return alpha_star

    @staticmethod
    def utility(I, gamma):
        return -I ** (1 - gamma) / (1 - gamma)

    def simulate_one_trial(self,
                           alpha_star: np.ndarray,
                           dB1: np.ndarray = None,
                           dB2: np.ndarray = None) -> SimulationResult:
        Ss = [self.S0]
        Xs = [self.X0]
        Is = [self.run_config.I0]
        n_steps = len(alpha_star)
        if dB1 is None:
            dB1 = self._simulate_dB(length=n_steps - 1)
        if dB2 is None:
            dB2 = self._simulate_dB(length=n_steps - 1)
        for i in range(1, n_steps):
            last_S = Ss[-1]
            last_X = Xs[-1]
            last_I = Is[-1]
            next_X = self.next_X(last_X=last_X, last_dB2=dB2[i - 1])
            next_S = self.next_S(last_S=last_S, last_I=last_I, last_dB1=dB1[i - 1])
            next_I = self.next_I(last_alpha=alpha_star[i], last_I=last_I, last_S=last_S, last_X=last_X,
                                 last_dB1=dB1[i - 1], last_dB2=dB2[i - 1])
            if next_I < 0:
                next_I = 0.0001
            if next_S is not None and next_S < 0:
                next_S = 0.0001
            Ss.append(next_S)
            Is.append(next_I)
            Xs.append(next_X)
        Is = np.array(Is)
        Ss = np.array(Ss)
        Xs = np.array(Xs)
        Utility = self.utility(I=Is, gamma=self.gamma)
        simulation_result = SimulationResult(
            Is=Is,
            Utility=Utility,
            Xs=Xs,
            Ss=Ss,
            dB1=dB1,
            dB2=dB2
        )
        simulation_result.Is = Is
        return simulation_result

    def run_monte_carlo_simulation(self,
                                   alpha_star: np.ndarray,
                                   dB1: np.ndarray = None,
                                   dB2: np.ndarray = None) -> Tuple[
        SimulationResult, List[SimulationResult]]:
        """
        Simulate future X and I from starting point
        """
        simulation_results = []
        if self.run_config.is_parallel:
            with Pool(self.run_config.cpu) as p:
                simulation_results = list(
                    p.map(partial(self.simulate_one_trial, dB1=dB1, dB2=dB2),
                          [alpha_star] * self.run_config.n_trials_monte_carlo_simulation),
                )
        else:
            for i in range(self.run_config.n_trials_monte_carlo_simulation):
                simulation_result = self.simulate_one_trial(
                    alpha_star=alpha_star,
                    dB1=dB1,
                    dB2=dB2
                )
                simulation_results.append(simulation_result)
        average_simulation_result = SimulationResult.average_simulation_results(simulation_results)
        return average_simulation_result, simulation_results

    def run_model_and_monte_carlo_simulation(
            self,
            Xs: np.ndarray,
            Ss: np.ndarray,
            Is: np.ndarray
    ) -> ModelResult:
        alpha_star = self.estimate_alpha(Xs, Ss, Is)
        alpha_star[alpha_star < 0] = 0

        if self.run_config.is_infer_dB_from_data_for_monte_carlo:
            dB1 = IFunctions.get_d_B1_from_data(Is=Is, Ss=Ss)
            dB2 = IFunctions.get_d_B2_from_data(S=Ss, X=Xs, Is=Is)
            assert len(dB1) == len(alpha_star) - 1
            assert len(dB2) == len(alpha_star) - 1
        else:
            dB1 = None
            dB2 = None

        average_simulation_result, all_simulation_results = self.run_monte_carlo_simulation(
            alpha_star=alpha_star,
            dB1=dB1,
            dB2=dB2
        )
        model_result = ModelResult(
            model_type=self.model_type,
            alpha_star=alpha_star,
            average_simulation_result=average_simulation_result,
            all_simulation_results=all_simulation_results
        )
        return model_result
