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
        for i in range(trial_length-1):
            new_sigma_s = conf.Ss[i+1]-conf.Ss[i]+ conf.beta * conf.Ss[i] * conf.Is[i] * conf.dt
            new_sigma_s = new_sigma_s/math.sqrt(conf.Ss[i] * conf.Is[i])/dB[i]
            sigma_s_lst.append(new_sigma_s)
        return sigma_s_lst

    @property
    def simulate_all_scenarios(self):
        sigma_s_all_lst = []
        for scenario in range(self.scenarios):
            sigma_s_lst = self.simulate_one_scenario
            sigma_s_lst_ = sum(sigma_s_lst)/len(sigma_s_lst)
            sigma_s_all_lst.append(sigma_s_lst_)
        return sigma_s_all_lst