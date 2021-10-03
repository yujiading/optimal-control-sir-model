import math
from abc import ABC, abstractmethod

import numpy as np

import library.models.model_params
from library.H_functions import HFunctions, SimpleHFunctions
from library.I_functions import IFunctions
from library.g_functions import GFunctions
from library.models import model_params


class BaseModelAlphaStar(ABC):
    def __init__(self, gamma, T=None):
        self.gamma = gamma
        self.T = T

    @abstractmethod
    def get_alpha_star(self, Xs, Ss, Is, length) -> np.ndarray:
        pass


class AlphaStarLowConst(BaseModelAlphaStar):
    def get_alpha_star(self, Xs, Ss, Is, length) -> np.ndarray:
        ret = (library.models.model_params.k1_bar - library.models.model_params.K0) \
              / library.models.model_params.sigma ** 2 / abs(self.gamma)
        alpha_star_ = [min(1, max(0, ret))] * length
        return np.array(alpha_star_)


class AlphaStarLowOU(BaseModelAlphaStar):
    def get_alpha_star(self, Xs, Ss, Is, length) -> np.ndarray:
        alpha_star_ = []
        length = len(Is)
        for t in range(length):
            A1 = HFunctions.A_1(gamma=self.gamma, tau=self.T - t * model_params.dt)
            A2 = HFunctions.A_2(gamma=self.gamma, tau=self.T - t * model_params.dt)
            X = Xs[t]
            alpha_star_.append((X - library.models.model_params.sigma_x * (
                    A1 * X + A2)) / self.gamma / library.models.model_params.sigma)
        return np.array(alpha_star_)


class AlphaStarModerateOU(BaseModelAlphaStar):
    def _A11_X_A12(self, t, X_t):
        return HFunctions.A_1(gamma=self.gamma, tau=self.T - t * model_params.dt) * X_t + HFunctions.A_2(
            gamma=self.gamma, tau=self.T - t * model_params.dt)

    def _get_alpha0(self, Xs, length):
        alpha0 = []
        for i in range(length):
            alpha0.append((Xs[i] - library.models.model_params.sigma_x * self._A11_X_A12(
                t=library.models.model_params.dt * i,
                X_t=Xs[i])) / self.gamma / library.models.model_params.sigma)
        return np.array(alpha0)

    @staticmethod
    def _get_Zs(X, Z_0, Ss, Is, length):
        # d logZ = a dt+b dB1+c dB2
        a = -library.models.model_params.mu + X ** 2 / 2 + library.models.model_params.beta ** 2 * Ss * Is / 2 / library.models.model_params.sigma_s ** 2
        b = -library.models.model_params.beta * (Ss * Is) ** 0.5 / library.models.model_params.sigma_s
        c = X

        logzs = np.zeros(length)
        logzs[0] = math.log(Z_0)
        dB1 = IFunctions.get_d_B1(length=length)
        dB2 = IFunctions.d_B2(length=length)
        for i in range(length - 1):
            if isinstance(c, np.ndarray):
                c = c[i]
            logzs[i + 1] = logzs[i] + library.models.model_params.dt * a[i] + b[i] * dB1[i] + c * dB2[i]
        zs = np.exp(logzs)
        return zs

    def _get_alpha1(self, Xs, Ss, Is, length):
        Z_0 = 1 / HFunctions.H(i=1, gamma=self.gamma, tau=self.T, X_t=0)
        zs = self._get_Zs(X=Xs, Z_0=Z_0, Ss=Ss, Is=Is, length=length)
        gfunctions = GFunctions(gamma=self.gamma, T=self.T)
        alpha1 = []
        for i in range(length):
            g = gfunctions.g(t=library.models.model_params.dt * i, X_t=Xs[i])
            a1 = zs[i] ** (1 / self.gamma) * Ss[i] / library.models.model_params.sigma / HFunctions.H(i=1,
                                                                                                      gamma=self.gamma,
                                                                                                      tau=self.T - library.models.model_params.dt * i,
                                                                                                      X_t=Xs[i])
            a2 = g * Xs[
                i] / self.gamma - library.models.model_params.sigma_x * gfunctions.d_g_d_X(
                t=library.models.model_params.dt * i, X_t=Xs[
                    i]) + library.models.model_params.sigma_x * g / self.gamma * self._A11_X_A12(
                t=library.models.model_params.dt * i,
                X_t=Xs[i])

            alpha1.append(a1 * a2)
        return np.array(alpha1)

    def get_alpha_star(self, Xs, Ss, Is, length) -> np.ndarray:
        alpha0 = self._get_alpha0(Xs=Xs, length=length)
        alpha1 = self._get_alpha1(Xs=Xs, Ss=Ss, Is=Is, length=length)
        return alpha0 + library.models.model_params.eps_moderate * alpha1 + library.models.model_params.eps_moderate ** 2


class AlphaStarModerateConst(BaseModelAlphaStar):
    @property
    def _alpha0(self):
        return library.models.model_params.X_bar / self.gamma / library.models.model_params.sigma

    def _get_g2(self, t):
        a1 = library.models.model_params.beta ** 2 / 2 / library.models.model_params.sigma_s ** 2
        h_2 = SimpleHFunctions(gamma=self.gamma, i=2).h(tau=self.T - t * model_params.dt)
        h_1 = SimpleHFunctions(gamma=self.gamma, i=1).h(tau=self.T - t * model_params.dt)
        a2 = h_2 - h_1 ** 2
        a3 = self.gamma * library.models.model_params.mu - library.models.model_params.X_bar ** 2 / self.gamma
        return a1 * a2 / a3

    def _get_alpha1(self, Ss, Is, length):
        z_0 = 1 / SimpleHFunctions(gamma=self.gamma, i=1).h(tau=self.T)
        zs = AlphaStarModerateOU._get_Zs(X=library.models.model_params.X_bar, Z_0=z_0, Ss=Ss, Is=Is, length=length)
        alpha1 = []
        for i in range(length):
            h_1 = SimpleHFunctions(gamma=self.gamma, i=1).h(tau=self.T - library.models.model_params.dt * i)
            a1 = zs[i] ** (1 / self.gamma) * Ss[i] / library.models.model_params.sigma / h_1
            a2 = self._get_g2(t=library.models.model_params.dt * i) * library.models.model_params.X_bar / self.gamma
            alpha1.append(a1 * a2)
        return np.array(alpha1)

    def get_alpha_star(self, Xs, Ss, Is, length) -> np.ndarray:
        alpha1 = self._get_alpha1(Ss=Ss, Is=Is, length=length)
        return self._alpha0 + library.models.model_params.eps_moderate * alpha1 + library.models.model_params.eps_moderate ** 2
