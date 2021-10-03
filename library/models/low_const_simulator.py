from model_result import ModelResult, SimulationResult
from model_mapper import ModelTypes
from base_simulator import BaseSimulator

class LowConstSimulator(BaseSimulator):
    model_type = ModelTypes.LowConst
    def estimate_alpha(self, Xs, Ss, Is, gamma, T):
        AlphaStarLowConst
        return

    def _simulate_Ss(self, ):
        self.next_S(last_I=0)

    def _simulate_Is(self, X0, I0, alpha):
        pass

    def run_simulation(self, alpha_star) -> SimulationResult:
        dB2 = self.simulate_dB2()
        Xs = self.simulate_Xs(dB2)
        Is = self.simulate_Is()
        simulation_result = SimulationResult()
        return simulation_result
