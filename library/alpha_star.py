import numpy as np
import math
from library import conf
from library.H_functions import HFunctions, SimpleHFunctions
from library.I_functions import IFunctions
from library.g_functions import GFunctions


class AlphaStarLowConst:
    def __init__(self, gamma, T=None):
        self.gamma = gamma

    @property
    def alpha_star(self):
        ret = (conf.k1_bar - conf.K0) / conf.sigma ** 2 / abs(self.gamma)
        return min(1, max(0, ret))


class AlphaStarLowOU:
    def __init__(self, T, gamma):
        self.gamma = gamma
        self.T = T

    @property
    def alpha_star(self):
        alpha_star_ = []
        for t in range(conf.length):
            A1 = HFunctions.A_1(gamma=self.gamma, tau=self.T - t)
            A2 = HFunctions.A_2(gamma=self.gamma, tau=self.T - t)
            X = conf.Xs[t]
            alpha_star_.append((X - conf.sigma_x * (A1 * X + A2)) / self.gamma / conf.sigma)
        return np.array(alpha_star_)


class AlphaStarModerateOU:
    def __init__(self, T, gamma):
        self.gamma = gamma
        self.T = T

    def _A11_X_A12(self, t, X_t):
        return HFunctions.A_1(gamma=self.gamma, tau=self.T - t) * X_t + HFunctions.A_2(gamma=self.gamma, tau=self.T - t)

    @property
    def alpha0(self):
        alpha0 = []
        for i in range(conf.length):
            alpha0.append((conf.Xs[i] - conf.sigma_x * self._A11_X_A12(t=conf.dt * i,
                                                                       X_t=conf.Xs[i])) / self.gamma / conf.sigma)
        return np.array(alpha0)

    @staticmethod
    def Zs(X, Z_0):

        # d logZ = a dt+b dB1+c dB2
        a = -conf.mu + X ** 2 / 2 + conf.beta ** 2 * conf.Ss * conf.Is / 2 / conf.sigma_s ** 2
        b = -conf.beta * (conf.Ss * conf.Is) ** 0.5 / conf.sigma_s
        c = X

        logzs = np.zeros(conf.length)
        logzs[0] = math.log(Z_0)
        dB1 = IFunctions.d_B1()
        dB2 = IFunctions.d_B2(S=conf.Ss, X=X)
        for i in range(conf.length - 1):
            if isinstance(c, np.ndarray):
                c = c[i]
            logzs[i + 1] = logzs[i] + conf.dt * a[i] + b[i] * dB1[i] + c * dB2[i]
        zs = np.exp(logzs)
        return zs

    @property
    def alpha1(self):
        Z_0 = 1 / HFunctions.H(i=1, gamma=self.gamma, tau=self.T, X_t=0)
        zs = self.Zs(X=conf.Xs, Z_0=Z_0)
        gfunctions = GFunctions(gamma=self.gamma, T=self.T)
        alpha1 = []
        for i in range(conf.length):
            g = gfunctions.g(t=conf.dt * i, X_t=conf.Xs[i])
            a1 = zs[i] ** (1 / self.gamma) * conf.Ss[i] / conf.sigma / HFunctions.H(i=1, gamma=self.gamma,
                                                                                    tau=self.T - conf.dt * i,
                                                                                    X_t=conf.Xs[i])
            a2 = g * conf.Xs[
                i] / self.gamma - conf.sigma_x * gfunctions.d_g_d_X(t=conf.dt * i, X_t=conf.Xs[
                i]) + conf.sigma_x * g / self.gamma * self._A11_X_A12(t=conf.dt * i,
                                                                      X_t=conf.Xs[i])

            alpha1.append(a1 * a2)
        return np.array(alpha1)

    @property
    def alpha_star(self):
        return self.alpha0 + conf.eps_moderate * self.alpha1 + conf.eps_moderate ** 2


class AlphaStarModerateConst:
    def __init__(self, T, gamma):
        self.gamma = gamma
        self.T = T

    @property
    def alpha0(self):
        return conf.X_bar / self.gamma / conf.sigma

    def g2(self, t):
        a1 = conf.beta ** 2 / 2 / conf.sigma_s ** 2
        h_2 = SimpleHFunctions(gamma=self.gamma, i=2).h(tau=self.T - t)
        h_1 = SimpleHFunctions(gamma=self.gamma, i=1).h(tau=self.T - t)
        a2 = h_2 - h_1 ** 2
        a3 = self.gamma * conf.mu - conf.X_bar ** 2 / self.gamma
        return a1 * a2 / a3

    @property
    def alpha1(self):
        z_0 = 1 / SimpleHFunctions(gamma=self.gamma, i=1).h(tau=self.T)
        zs = AlphaStarModerateOU.Zs(X=conf.X_bar, Z_0=z_0)
        alpha1 = []
        for i in range(conf.length):
            h_1 = SimpleHFunctions(gamma=self.gamma, i=1).h(tau=self.T - conf.dt * i)
            a1 = zs[i] ** (1 / self.gamma) * conf.Ss[i] / conf.sigma / h_1
            a2 = self.g2(t=conf.dt * i) * conf.X_bar / self.gamma
            alpha1.append(a1 * a2)
        return np.array(alpha1)

    @property
    def alpha_star(self):
        return self.alpha0 + conf.eps_moderate * self.alpha1 + conf.eps_moderate ** 2
