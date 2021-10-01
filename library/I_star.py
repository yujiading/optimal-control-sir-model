from typing import Union

import numpy as np

from library import conf
from library.I_functions import IFunctions


class IStarLowConst:
    def __init__(self, alpha_star: Union[float, np.array]):
        self.alpha_star = alpha_star  # estimated alpha

    def I_star_(self, S, X, eps):
        Imudt1 = IFunctions.I_mu(alpha=self.alpha_star, S=S, X=X) * conf.dt + 1
        Isig1 = IFunctions.I_sigma_1()
        Isig2 = IFunctions.I_sigma_2(alpha=self.alpha_star)
        I = [eps]
        dB2 = IFunctions.d_B2(S=S, X=X)

        for i in range(conf.length - 1):
            if isinstance(Imudt1, np.ndarray):
                Imudt1_ = Imudt1[i]
            else:
                Imudt1_ = Imudt1
            if isinstance(Isig2, np.ndarray):
                Isig2_ = Isig2[i]
            else:
                Isig2_ = Isig2
            ret = Imudt1_ + Isig2_ * dB2[i]
            ret = ret * I[-1]
            if not isinstance(S, int):
                dB1 = IFunctions.d_B1()
                ret = ret + Isig1 * (S[i] * I[-1]) ** 0.5 * dB1[i]
                if ret <= 0:
                    ret = 0.001
            I.append(ret)
        return np.array(I)

    @property
    def I_star(self):
        return self.I_star_(S=1, X=conf.X_bar, eps=conf.eps_low)


class IStarLowOU(IStarLowConst):
    def __init__(self, alpha_star: Union[float, np.array]):
        super().__init__(alpha_star)

    @property
    def I_star(self):
        return self.I_star_(S=1, X=conf.Xs, eps=conf.eps_low)


class IStarModerateOU(IStarLowConst):
    def __init__(self, alpha_star: Union[float, np.array]):
        super().__init__(alpha_star)

    @property
    def I_star(self):
        return self.I_star_(S=conf.Ss, X=conf.Xs, eps=conf.eps_moderate)


class IStarModerateConst(IStarLowConst):
    def __init__(self, alpha_star: Union[float, np.array]):
        super().__init__(alpha_star)

    @property
    def I_star(self):
        return self.I_star_(S=conf.Ss, X=conf.X_bar, eps=conf.eps_moderate)

# class IStarLowOU:
#     def __init__(self, alpha_star: Union[float, np.array], eps: float, T, gamma):
#         self.eps = eps  # initial I
#         self.alpha_star = alpha_star  # estimated alpha
#         self.T = T
#         self.gamma = gamma
#
#     @property
#     def Z(self):
#         dB2 = IFunctions.d_B2(S=1, X=conf.Xs)
#         rhs = (conf.r + conf.Xs[:-1] ** 2) * conf.dt + conf.Xs[:-1] * dB2
#         H_0_initial = HFunctions.H(i=0, gamma=self.gamma, tau=self.T, X_t=conf.Xs[0])
#         Z_ = [(self.eps / H_0_initial) ** self.gamma]
#         for i in range(conf.length - 1):
#             Z_.append((rhs[i] + 1) * Z_[-1])
#         return np.array(Z_)
#
#     @property
#     def I_star(self):
#         Zs = self.Z
#         I = [self.eps]
#         for i in range(conf.length - 1):
#             a = conf.Xs[i] * Zs[i] / (Zs[i + 1] - Zs[i])
#             b = conf.sigma_x / conf.dXs[i]
#             if isinstance(self.alpha_star, np.ndarray):
#                 alpha = self.alpha_star[i]
#             else:
#                 alpha = self.alpha_star
#             I.append((alpha * conf.sigma + a - b) / (a - b) * I[i - 1])
#         return np.array(I)
