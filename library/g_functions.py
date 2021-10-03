import math

import numpy as np
import scipy.integrate as integrate

import library.models.model_params
from library import conf
from library.H_functions import HFunctions


class GFunctions:
    def __init__(self, T, gamma):
        self.gamma = gamma
        self.T = T

    def _M(self, t, tau):
        a1 = (HFunctions.b_2(gamma=self.gamma / 2) - HFunctions.b_2(gamma=self.gamma) -
              HFunctions.theta(gamma=self.gamma)) * (tau - t) / 2
        a2 = 2 * HFunctions.theta(gamma=self.gamma) / (HFunctions._bottom(tau=tau - t, gamma=self.gamma))

        return math.exp(a1) * a2

        # return a1 + cmath.log(a2).real

    #     def _V_Y(self, ts, tau, X_ts):
    #         integral = integrate.quad(
    #             lambda x: math.exp(2 * self._M(ts=x, tau=tau)), ts, tau)
    #         print('V_Y has approx. error: ', integral[1])
    #         return self.sigma_x**2 * integral[0]
    def _V_Y(self, t, tau, X_t):
        a1 = HFunctions.b_2(gamma=self.gamma / 2) - HFunctions.b_2(gamma=self.gamma) - HFunctions.theta(
            gamma=self.gamma)
        a2 = HFunctions.theta(gamma=self.gamma) - HFunctions.b_2(gamma=self.gamma)
        a3 = HFunctions.b_2(gamma=self.gamma) + HFunctions.theta(gamma=self.gamma)
        low = 0.001
        if tau - t <= low:
            low = (tau - t) / 10
        integral = integrate.quad(
            lambda x: math.exp(a1 * x) / (a2 + a3 * math.exp(-HFunctions.theta(gamma=self.gamma) * x)) ** 2, low,
            tau - t)

        #         print('V_Y has approx. error: ', integral[1])
        return library.models.model_params.sigma_x ** 2 * 4 * HFunctions.theta(gamma=self.gamma) ** 2 * integral[0]

    def _m_Y(self, t, tau, X_t):
        a2 = X_t * self._M(t, tau) + integrate.quad(
            lambda x: self._M(x, tau) *
                      (
                                  library.models.model_params.lambda_x * library.models.model_params.X_bar + library.models.model_params.sigma_x ** 2 / self.gamma * HFunctions.A_2(tau=tau - x,
                                                                                                                                                                                    gamma=self.gamma)),
            t, tau)[0]
        a3 = HFunctions.A_2(tau=self.T - tau, gamma=self.gamma) / HFunctions.A_1(tau=self.T -
                                                                                     tau, gamma=self.gamma)
        return a2 + a3

    def _integrant_g(self, x, t, X_t):
        if 1 - 2 * self._V_Y(t=t, tau=x, X_t=X_t) * HFunctions.A_1(
                tau=self.T - x, gamma=self.gamma) / self.gamma > 0:
            a1 = HFunctions.H(i=2, gamma=self.gamma, tau=x - t,
                              X_t=X_t) / 2 * library.models.model_params.beta ** 2 / library.models.model_params.sigma_s ** 2 / self.gamma / (
                         1 - 2 * self._V_Y(t=t, tau=x, X_t=X_t) *
                         HFunctions.A_1(tau=self.T - x, gamma=self.gamma) / self.gamma) ** 0.5
            a2 = math.exp(2 * HFunctions.A_3(tau=self.T - x, gamma=self.gamma) / self.gamma -
                          (HFunctions.A_2(tau=self.T - x, gamma=self.gamma)) ** 2 / self.gamma /
                          HFunctions.A_1(tau=self.T - x, gamma=self.gamma) +
                          (self._m_Y(t=t, tau=x, X_t=X_t)) ** 2 *
                          HFunctions.A_1(tau=self.T - x, gamma=self.gamma) /
                          (self.gamma - 2 * self._V_Y(t=t, tau=x, X_t=X_t) *
                           HFunctions.A_1(tau=self.T - x, gamma=self.gamma)))
        else:
            a1 = HFunctions.H(i=2, gamma=self.gamma,
                              tau=x - t,
                              X_t=X_t) / 2 * library.models.model_params.beta ** 2 / library.models.model_params.sigma_s ** 2 / self.gamma / (
                         2 * math.pi * self._V_Y(t=t, tau=x, X_t=X_t)) ** 0.5

            coef1 = round(HFunctions.A_1(tau=self.T - x, gamma=self.gamma) / self.gamma, 4)
            coef2 = round(self._m_Y(t=t, tau=x, X_t=X_t), 4)
            coef3 = round(2 * self._V_Y(t=t, tau=x, X_t=X_t), 4)

            a2 = math.exp(
                2 * HFunctions.A_3(tau=self.T - x, gamma=self.gamma) / self.gamma -
                (HFunctions.A_2(tau=self.T - x, gamma=self.gamma)) ** 2 / self.gamma /
                HFunctions.A_1(tau=self.T - x, gamma=self.gamma)) * integrate.quad(
                lambda y: math.exp(
                    coef1 * y ** 2 -
                    (y - coef2) ** 2 / coef3), -np.inf, np.inf)[0]

        return a1 * a2

    def g(self, t, X_t):
        return integrate.quad(lambda x: self._integrant_g(x=x, t=t, X_t=X_t),
                              t, self.T)[0]

    def _integrant_d_g_d_X(self, x, t, X_t):
        a1 = (HFunctions.A_1(tau=x - t, gamma=self.gamma / 2) * X_t +
              HFunctions.A_2(tau=x - t, gamma=self.gamma / 2)) / self.gamma * 2
        a2 = 2 * self._m_Y(t=t, tau=x, X_t=X_t) * HFunctions.A_1(
            tau=self.T -
                x, gamma=self.gamma) / (self.gamma - 2 * self._V_Y(t=t, tau=x, X_t=X_t) *
                                        HFunctions.A_1(tau=self.T - x, gamma=self.gamma)) * math.exp(self._M(t, x))
        return self._integrant_g(x=x, t=t, X_t=X_t) * (a1 + a2)

    def d_g_d_X(self, t, X_t):
        return integrate.quad(
            lambda x: self._integrant_d_g_d_X(x=x, t=t, X_t=X_t), t, self.T)[0]
