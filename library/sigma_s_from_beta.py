import math
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

import library.models.model_params
from library import conf


class SigmaSFromBeta:
    """
    S(t+dt) = S(t) - (beta + a) * S(t) * I(t) * dt
    S(0) = 1
    """

    def __init__(self,
                 beta_change_low: float = -0.01,
                 beta_change_up: float = 0.01,
                 trial_length: int = 10,
                 scenarios: int = 1000,
                 S_0: float = 1):
        self.trial_length = trial_length
        self.beta_change_low = beta_change_low
        self.beta_change_up = beta_change_up
        self.scenarios = scenarios
        self.S_0 = S_0

    def monte_carlo_one_scenario(self, beta_change) -> List:
        S_series = [self.S_0]
        for i in range(self.trial_length):
            last_S = S_series[-1]
            new_S = last_S - (library.models.model_params.beta + beta_change) * last_S * conf.Is[i] * library.models.model_params.dt
            S_series.append(new_S)
        return S_series

    @property
    def monte_carlo_all_scenarios(self) -> List[List]:
        beta_change_step = (self.beta_change_up - self.beta_change_low) / 1000
        beta_changes = list(np.arange(start=self.beta_change_low, stop=self.beta_change_up, step=beta_change_step))
        beta_changes.append(self.beta_change_up)
        S_all_series = []
        for beta_change in beta_changes:
            S_series = self.monte_carlo_one_scenario(beta_change=beta_change)
            S_all_series.append(S_series)
        return S_all_series

    def plot_all_scenarios(self, S_all_series: List[List]):
        for S_series in S_all_series:
            plt.plot(S_series, 'gray')
        # S_series_0 = self.monte_carlo_one_scenario(beta_change=0)
        # plt.plot(S_series_0, 'g')
        S_series_up = self.monte_carlo_one_scenario(beta_change=self.beta_change_up)
        plt.plot(S_series_up, 'b')
        S_series_low = self.monte_carlo_one_scenario(beta_change=self.beta_change_low)
        plt.plot(S_series_low, 'r')
        plt.show()

    @staticmethod
    def get_item_from_each_sublist(list_of_list: List[List], index: int):
        lst = [item[index] for item in list_of_list]
        return lst

    @staticmethod
    def get_std_of_normal_given_prob(low: float, up: float, mean: float, percent: float):
        """
        given P(low<X<up)=percent and assume X is normal with mean, find std
        """

        # func = lambda sigma: norm.cdf((up-mean)/sigma)-norm.cdf((low-mean)/sigma)-percent
        # std = fsolve(func, [0.01])[0]
        std = (up - mean) / norm.ppf(0.5 + percent / 2)
        return std

    def sigma_s_one_estimate(self, S_all_series: List[List], trail_index: int, percent: float):
        """
        trail_index: 1 to self.trail_length
        """
        S_series = self.get_item_from_each_sublist(list_of_list=S_all_series, index=trail_index)
        # S_series_mean = sum(S_series)/len(S_series)
        S_series_up = max(S_series)
        S_series_low = min(S_series)
        S_series_mean = (S_series_up + S_series_low) / 2
        sigma_s_ = self.get_std_of_normal_given_prob(low=S_series_low, up=S_series_up, mean=S_series_mean,
                                                     percent=percent)
        sigma_s_ = sigma_s_ / math.sqrt(S_series_mean * conf.Is[trail_index])
        return sigma_s_
