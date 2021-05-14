from abc import ABC, abstractmethod

from library import conf


class AlphaStar(ABC):
    def __init__(self, gamma):
        self.gamma = gamma

    @abstractmethod
    def alpha_star(self):
        pass


class AlphaStarLowConst(AlphaStar):
    @property
    def alpha_star(self):
        ret = (conf.k1_bar - conf.K0) / conf.sigma ** 2 / abs(self.gamma)
        return min(1, max(0, ret))
