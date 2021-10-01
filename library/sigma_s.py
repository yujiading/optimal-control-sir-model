import math

import numpy as np
from library import conf


class SigmaSDoubleSum:
    def __init__(self, scenarios: int = 10000):
        self.scenarios = scenarios
    @property
    def simulate_one_scenario(self):
        """
        sigma_s = (S(t+dt) - S(t) + beta * S(t) * I(t) * dt)/sqrt(S(t)I(t))/dB
        dB = N(0, sqrt(dt))
        """

        trial_length = len(conf.Is)
        dB = np.random.randn(trial_length) * math.sqrt(conf.dt)
        sigma_s_lst = []
        sigma_s_square_lst = []
        for i in range(trial_length-1):
            new_sigma_s = conf.Ss[i+1]-conf.Ss[i]+ conf.beta * conf.Ss[i] * conf.Is[i] * conf.dt
            new_sigma_s = new_sigma_s/math.sqrt(conf.Ss[i] * conf.Is[i])/dB[i]
            new_sigma_s_square = new_sigma_s ** 2
            sigma_s_square_lst.append(new_sigma_s_square)
            sigma_s_lst.append(new_sigma_s)
        return sigma_s_lst, sigma_s_square_lst

    @property
    def simulate_all_scenarios(self):
        sigma_s_lst = []
        sigma_s_square_lst=[]
        for scenario in range(self.scenarios):
            sigma_s_lst_, sigma_s_square_lst_ = self.simulate_one_scenario
            sigma_s_ = sum(sigma_s_lst_)/len(sigma_s_lst_)
            sigma_s_lst.append(sigma_s_)

            sigma_s_square_ = sum(sigma_s_square_lst_) / len(sigma_s_square_lst_)
            sigma_s_square_lst.append(sigma_s_square_)
        return sigma_s_lst, sigma_s_square_lst


class SigmaSSquareConst:

    @property
    def sigma_s(self):
        """
        sigma_s^2 = mean (S(t+dt) - S(t) + beta * S(t) * I(t) * dt)/sqrt(S(t)I(t)dt)

        """
        trial_length = len(conf.Is)
        sigma_s_square_lst = []
        for i in range(trial_length - 1):
            new_sigma_s = conf.Ss[i + 1] - conf.Ss[i] + conf.beta * conf.Ss[i] * conf.Is[i] * conf.dt
            new_sigma_s = new_sigma_s / math.sqrt(conf.Ss[i] * conf.Is[i]*conf.dt)
            new_sigma_s_square = new_sigma_s**2
            sigma_s_square_lst.append(new_sigma_s_square)
        return math.sqrt(np.sum(sigma_s_square_lst)/trial_length)
