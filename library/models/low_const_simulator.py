from model_result import ModelResult, SimulationResult
from model_mapper import ModelTypes
from base_simulator import BaseSimulator
from library.alpha_star import AlphaStarLowConst


class LowConstSimulator(BaseSimulator):
    model_type = ModelTypes.LowConst


    def __init__(self, gamma, T):
        super().__init__(gamma=gamma)
        self.T = T

    def estimate_alpha(self, Xs, Ss, Is):
        alpha_star_model = AlphaStarLowConst(gamma=self.gamma, T=self.T)
        alpha_star = alpha_star_model.get_alpha_star(Xs=Xs, Ss=Ss, Is=Is)
        return alpha_star

    def _simulate_Is(self, X0, I0, alpha):
        pass

    def run_monte_carlo_simulation(self, alpha_star) -> SimulationResult:
        dB2 = self.simulate_dB2()
        Xs = self.simulate_Xs(dB2)
        Is = self.simulate_Is()
        simulation_result = SimulationResult()
        return simulation_result
