import math

import scipy.integrate as integrate

from library import conf


class HFunctions:

    @staticmethod
    def b_1(gamma):
        return (1 - gamma) / gamma

    @staticmethod
    def b_2(gamma):
        ret = (gamma - 1) / gamma
        ret = ret * conf.sigma_x
        ret = ret - conf.lambda_x
        return 2 * ret

    @staticmethod
    def b_3(gamma):
        return conf.sigma_x ** 2 / gamma

    @staticmethod
    def theta(gamma):
        b2 = HFunctions.b_2(gamma=gamma)
        b1 = HFunctions.b_1(gamma=gamma)
        b3 = HFunctions.b_3(gamma=gamma)
        return (b2 ** 2 - 4 * b1 * b3) ** 0.5

    @staticmethod
    def _exp(n, tau, gamma):
        return 1 - math.exp(-HFunctions.theta(gamma=gamma) * tau / n)

    @staticmethod
    def _bottom(tau, gamma):
        theta = HFunctions.theta(gamma=gamma)
        b2 = HFunctions.b_2(gamma=gamma)
        exp_ = HFunctions._exp(n=1, tau=tau, gamma=gamma)
        return 2 * theta - (b2 + theta) * exp_

    @staticmethod
    def A_1(tau, gamma):
        b1 = HFunctions.b_1(gamma=gamma)
        exp_ = HFunctions._exp(n=1, tau=tau, gamma=gamma)
        bottom = HFunctions._bottom(tau=tau, gamma=gamma)
        return (2 * b1 * exp_) / bottom

    @staticmethod
    def A_2(tau, gamma):
        b1 = HFunctions.b_1(gamma=gamma)
        exp_2 = HFunctions._exp(n=2, tau=tau, gamma=gamma)
        theta = HFunctions.theta(gamma=gamma)
        bottom = HFunctions._bottom(tau=tau, gamma=gamma)
        return (4 * conf.lambda_x * conf.X_bar * b1 * exp_2 ** 2) / theta / bottom

    @staticmethod
    def A_3_numeric(tau, gamma):
        coef2 = conf.sigma_x ** 2 / 2
        coef1 = coef2 / gamma + conf.lambda_x * conf.X_bar
        coef3 = (gamma - 1) * conf.mu * tau
        low = 0.001
        if tau <= low:
            low = tau / 10
        integral = integrate.quad(
            lambda x: coef1 * HFunctions.A_2(x, gamma=gamma) ** 2 + coef2 * HFunctions.A_1(x, gamma=gamma), low, tau)
        return integral[0] + coef3  # , integral[1]

    @staticmethod
    def A_3(tau, gamma):
        b2 = HFunctions.b_2(gamma=gamma)
        b1 = HFunctions.b_1(gamma=gamma)
        b3 = HFunctions.b_3(gamma=gamma)
        A1 = HFunctions.A_1(tau=tau, gamma=gamma)
        A2 = HFunctions.A_2(tau=tau, gamma=gamma)
        theta = HFunctions.theta(gamma=gamma)
        bottom = HFunctions._bottom(tau=tau, gamma=gamma)
        coef2 = conf.sigma_x ** 2 / 2
        coef1 = coef2 / gamma + conf.lambda_x * conf.X_bar
        coef3 = (gamma - 1) * conf.mu * tau
        a1 = 2 * conf.lambda_x * conf.X_bar * b2 * A2 / theta ** 3 / b3
        a2 = 2 * conf.lambda_x ** 2 * conf.X_bar ** 2 / theta ** 3
        a3 = -A1 / b3
        a4 = 8 * b1 ** 2 * tau / (b2 - theta) / (b1 * b3) ** 0.5
        # a5 = cmath.log(self._bottom(tau=tau) / 2 * self.theta)
        a5 = math.log(bottom / 2 * theta)
        a6 = b2 * (theta - 2 * (b1 * b3) ** 0.5) / b3 ** 2 / theta
        a7 = (b2 - 2 * (b1 * b3) ** 0.5) / theta
        a8 = (2 * b2 + 4 * (b1 * b3) ** 0.5 * math.exp(-theta * tau / 2)) / bottom
        a9 = -(b2 + theta) * A1 / 2 / b1
        a10 = 4 * tau * b1 ** 2 / (b2 + theta) ** 2
        a11 = 2 * theta * math.exp(-theta * tau) / bottom
        a12 = -2 * b1 * tau / (b2 + theta)

        result = coef1 * (a1 + a2 * (a3 + a4 * a5 + a6 * math.log(a7 * (a8 + a9)) + a10)) + coef2 * (
                math.log(a11) / b3 + a12) + coef3
        return result

        # result = coef1 * (a1 + a2 * (a3 + a4 * a5 + a6 * cmath.log(a7 * (a8 + a9)) + a10)) + coef2 * (
        #         cmath.log(a11) / self.b_3 + a12) + coef3
        # return result.real

    @staticmethod
    def H(i, gamma, tau, X_t):
        const = 0
        if i > 0:
            gamma = gamma / i
        if i == 0:
            i = 1
            const = (1 - gamma) * (conf.mu + conf.r) * tau
        A1 = HFunctions.A_1(tau=tau, gamma=gamma)
        A2 = HFunctions.A_2(tau=tau, gamma=gamma)
        A3 = HFunctions.A_3(tau=tau, gamma=gamma)
        power = A1 * X_t ** 2/2 + A2 * X_t + A3 + const
        power = power * i / gamma
        return math.exp(power)


class SimpleHFunctions:
    def __init__(self, i, gamma):
        self.i = i
        self.gamma = gamma

    @property
    def _gamma_over(self):
        return self.gamma / (2 ** (self.i - 1))

    @property
    def a_1(self):
        return (1 - self._gamma_over) / self._gamma_over

    @property
    def a_2(self):
        return (self._gamma_over - 1) * conf.mu

    def h(self, tau):
        power = 1 / self._gamma_over * (self.a_1 * conf.X_bar ** 2 / 2 + self.a_2) * tau
        return math.exp(power)
