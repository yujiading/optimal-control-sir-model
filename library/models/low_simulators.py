from library.models.base_simulator import BaseSimulator
from library.alpha_star import AlphaStarLowOU, AlphaStarLowConst
from library.models import model_params
from library.run_config import RunConfig
from library.models.model_mapper import ModelTypes


class LowOUSimulator(BaseSimulator):
    model_type = ModelTypes.LowOU
    alpha_star_model_class = AlphaStarLowOU

    def __init__(self, gamma, run_config: RunConfig):
        super().__init__(gamma=gamma, run_config=run_config)
        self.S0 = None

    @staticmethod
    def next_S(last_S, last_I, last_dB1):
        return None

    @staticmethod
    def next_I(last_alpha, last_I, last_S, last_X, last_dB1, last_dB2):
        ret = last_I + (
                model_params.r + last_alpha * model_params.sigma * last_X) * last_I * model_params.dt \
              + last_alpha * last_I * model_params.sigma * last_dB2
        return ret


class LowConstSimulator(LowOUSimulator):
    model_type = ModelTypes.LowConst
    alpha_star_model_class = AlphaStarLowConst

    def __init__(self, gamma, run_config: RunConfig):
        super().__init__(gamma=gamma, run_config=run_config)
        self.X0 = model_params.X_bar

    def next_X(self, last_X, last_dB2):
        return model_params.X_bar
