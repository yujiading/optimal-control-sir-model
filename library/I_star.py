from abc import ABC, abstractmethod

import numpy as np

from library import conf


class IStar(ABC):
    """
    dI = I_mu I dt + I_sigma_2 I dB2 + ...
    """

    def __init__(self, alpha_star, eps):
        self.eps = eps
        self.alpha_star = alpha_star

    def I_mu(self, alpha, S, X):
        return conf.beta * S - conf.mu + alpha * conf.sigma * X

    def I_sigma_2(self, alpha):
        return alpha * conf.sigma

    @property
    def d_B1(self):
        SI = conf.Is[:-1] * conf.Ss[:-1]
        return (conf.dSs + conf.beta * SI * conf.dt) / conf.sigma_s / (SI) ** 0.5

    def d_B2(self, S, X):
        if S == 1:
            item = 0
        else:
            item = conf.sigma_s * (conf.Is[:-1] * conf.Ss[:-1]) ** 0.5 * self.d_B1
        return (conf.dIs - self.I_mu(alpha=conf.alpha_fix, S=S, X=X) * conf.Is[:-1] * conf.dt + item) / self.I_sigma_2(
            alpha=conf.alpha_fix)

    @abstractmethod
    def I_star(self):
        pass


class IStarLowConst(IStar):
    @property
    def I_star(self):
        Imudt1 = self.I_mu(alpha=self.alpha_star, S=1, X=conf.X_bar) * conf.dt + 1
        Isig = self.I_sigma_2(alpha=self.alpha_star)
        I = [self.eps]
        for i in range(conf.length - 1):
            ret = Imudt1 + Isig * self.d_B2(S=1, X=conf.X_bar)[i]
            ret = ret * I[-1]
            I.append(ret)
        return np.array(I)
