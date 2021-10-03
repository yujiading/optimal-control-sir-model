from model_result import ModelResult, SimulationResult
from model_mapper import ModelTypes
from base_simulator import BaseSimulator
from library.alpha_star import AlphaStarLowOU


class LowOUSimulator(BaseSimulator):
    model_type = ModelTypes.LowOU
    alpha_star_model_class = AlphaStarLowOU

    def _simulate_Is(self, X0, I0, alpha):
        pass

    def simulate_one_trial(self, X0, I0, alpha_star) -> SimulationResult:
        dB2 = self._simulate_dB(length=self.run_config.n_steps_simulated_data_generation - 1)
        simulation_result = SimulationResult()
        return simulation_result
