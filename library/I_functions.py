from typing import Union
import math
import numpy as np

from library import conf


class IFunctions:
    """
       dI = I_mu I dt + I_sigma_2 I dB2 + I_sigma_1 sqrt(SI) dB1
    """

    @staticmethod
    def I_mu(alpha: Union[float, np.array], S: Union[float, np.array], X: Union[float, np.array]):
        return conf.beta * S - conf.mu + alpha * conf.sigma * X

    @staticmethod
    def I_sigma_2(alpha: Union[float, np.array]):
        return alpha * conf.sigma

    @staticmethod
    def I_sigma_1():
        return -conf.sigma_s

    @staticmethod
    def d_B1():
        return np.random.normal(loc=0, scale=math.sqrt(conf.dt), size=conf.length-1)

    @staticmethod
    def d_B2(S: Union[float, np.array], X: Union[float, np.array]):
        # np.random.seed(0)
        return np.random.normal(loc=0, scale=math.sqrt(conf.dt), size=conf.length-1)
    #
    # @staticmethod
    # def d_B1():
    #     SI = conf.Is[1:] * conf.Ss[1:]
    #     return (conf.dSs + conf.beta * SI * conf.dt) / conf.sigma_s / (SI) ** 0.5
    #
    # @staticmethod
    # def d_B2(S: Union[float, np.array], X: Union[float, np.array]):
    #     if isinstance(S, int):
    #         item = 0
    #     else:
    #         item = -IFunctions.I_sigma_1() * (conf.Is[:-1] * conf.Ss[:-1]) ** 0.5 * IFunctions.d_B1()
    #
    #     Imu = IFunctions.I_mu(alpha=conf.alpha_fix, S=S, X=X)
    #     if isinstance(Imu, np.ndarray):
    #         Imu = Imu[:-1]
    #     top = conf.dIs - Imu * conf.Is[:-1] * conf.dt + item
    #     bottom = IFunctions.I_sigma_2(alpha=conf.alpha_fix)
    #     return top / bottom
