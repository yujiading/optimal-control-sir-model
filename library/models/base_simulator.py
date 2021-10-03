from abc import ABC, abstractmethod
from model_result import ModelResult, SimulationResult


class BaseSimulator(ABC):
    """
    Run simulation for 1 infection rate & fix/changing X combination
    """
    model_type: str

    def __init__(self, gamma):
        self.gamma = gamma

    @staticmethod
    def utility(I, gamma):
        return -I ** (1 - gamma) / (1 - gamma)

    @abstractmethod
    def estimate_alpha(self, Xs, Ss, Is, gamma, T):
        pass

    @abstractmethod
    def simulate_one_trial(self, alpha_star) -> SimulationResult:
        pass

    def _simulate_Xs(self, ):
        pass

    def _simulate_dB(self):
        pass

    def run_simulation(self, alpha_star) -> SimulationResult:
        pass

    def run_model_and_simulation(self, Xs, Ss, Is) -> ModelResult:
        alpha_star = self.estimate_alpha(Xs, Ss, Is)
        simulation_result = self.run_simulation(alpha_star=alpha_star)
        model_result = ModelResult(
            model_type=self.model_type,
            alpha_star=alpha_star,
            simulation_result=simulation_result
        )
        return model_result
