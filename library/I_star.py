from typing import Union
import math
import numpy as np

import library.models.model_params
from library import conf
from library.I_functions import IFunctions


class IStarLowConst:
    def __init__(self, alpha_star: Union[float, np.array]):
        self.alpha_star = alpha_star  # estimated alpha

    def _I_star(self, S, X, eps, length):
        """
            dI = I_mu I dt + I_sigma_2 I dB2 + I_sigma_1 sqrt(SI) dB1
        """
        Imudt1 = IFunctions.I_mu(alpha=self.alpha_star, S=S, X=X) * library.models.model_params.dt + 1
        Isig1 = IFunctions.I_sigma_1()
        Isig2 = IFunctions.I_sigma_2(alpha=self.alpha_star)
        I = [eps]
        dB2 = IFunctions.d_B2(length=length)
        # print(f'dB2 {dB2}')

        for i in range(length - 1):
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
                dB1 = IFunctions.get_d_B1(length=length)
                ret = ret + Isig1 * math.sqrt(S[i] * I[-1]) * dB1[i]
                if ret <= 0:
                    ret = 0
            I.append(ret)
        return np.array(I)

    def get_I_star(self, Xs, Ss):
        length = len(Xs)
        return self._I_star(
            S=1,
            X=library.models.model_params.X_bar,
            eps=library.models.model_params.eps_low,
            length=length
        )


class IStarLowOU(IStarLowConst):
    def __init__(self, alpha_star: Union[float, np.array]):
        super().__init__(alpha_star)

    def get_I_star(self, Xs, Ss):
        length = len(Xs)
        return self._I_star(
            S=1,
            X=Xs,
            eps=library.models.model_params.eps_low,
            length=length
        )


class IStarModerateOU(IStarLowConst):
    def __init__(self, alpha_star: Union[float, np.array]):
        super().__init__(alpha_star)

    def get_I_star(self, Xs, Ss):
        length = len(Xs)
        return self._I_star(
            S=Ss,
            X=Xs,
            eps=library.models.model_params.eps_moderate,
            length=length
        )


class IStarModerateConst(IStarLowConst):
    def __init__(self, alpha_star: Union[float, np.array]):
        super().__init__(alpha_star)

    def get_I_star(self, Xs, Ss):
        length = len(Xs)
        return self._I_star(
            S=Ss,
            X=library.models.model_params.X_bar,
            eps=library.models.model_params.eps_moderate,
            length=length
        )

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
