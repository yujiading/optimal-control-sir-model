import math

from library.models.base_simulator import BaseSimulator
from library.alpha_star import AlphaStarModerateOU, AlphaStarModerateConst
from library.models import model_params
from library.run_config import RunConfig
from library.models.model_mapper import ModelTypes


class ModerateOUSimulator(BaseSimulator):
    model_type = ModelTypes.ModerateOU
    alpha_star_model_class = AlphaStarModerateOU

    @staticmethod
    def next_S(last_S, last_I, last_dB1):
        ret = last_S - model_params.beta * last_S * last_I * model_params.dt + model_params.sigma_s * math.sqrt(
            last_S * last_I) * last_dB1
        return ret

    @staticmethod
    def next_I(last_alpha, last_I, last_S, last_X, last_dB1, last_dB2):
        ret = last_I + (
                model_params.beta * last_S - model_params.mu + last_alpha * model_params.sigma * last_X) * last_I * model_params.dt \
              + last_alpha * last_I * model_params.sigma * last_dB2 - model_params.sigma_s * math.sqrt(
            last_S * last_I) * last_dB1
        return ret


class ModerateConstSimulator(ModerateOUSimulator):
    model_type = ModelTypes.ModerateConst
    alpha_star_model_class = AlphaStarModerateConst

    def __init__(self, gamma, run_config: RunConfig):
        super().__init__(gamma=gamma, run_config=run_config)
        self.X0 = model_params.X_bar

    def next_X(self, last_X, last_dB2):
        return model_params.X_bar
