import numpy as np
from library import conf
import math


class DataModerateOU:
    def __init__(self, I0, X0, S0, n_steps: int = 20, n_trials: int = 10000):
        self.X0 = X0
        self.I0 = I0
        self.S0 = S0
        self.n_trials = n_trials
        self.n_steps = n_steps
        self.d_B1_trials = None
        self.d_B2_trials = None
        self.Xs_trials = None
        self.Is_trials = None
        self.Ss_trials = None
        self.get_data()

    def next_X(self, last_X, last_dB2):
        ret = last_X + conf.lambda_x * (conf.X_bar - last_X) * conf.dt - conf.sigma_x * last_dB2
        return ret[0]

    def next_S(self, last_S, last_I, last_dB1):
        ret = last_S - conf.beta * last_S * last_I * conf.dt + conf.sigma_s * math.sqrt(last_S * last_I) * last_dB1
        return ret[0]
    def next_I(self, last_I, last_S, last_X, last_dB1, last_dB2):
        ret= last_I + (conf.beta * last_S - conf.mu + conf.alpha_fix * conf.sigma * last_X) * last_I * conf.dt \
               + conf.alpha_fix * last_I * conf.sigma * last_dB2 - conf.sigma_s * math.sqrt(last_S * last_I) * last_dB1
        return ret[0]
    @property
    def get_data_one_trial(self):
        Ss = [self.S0]
        Xs = [self.X0]
        Is = [self.I0]
        dB1 = []
        dB2 = []
        for i in range(1, self.n_steps):
            last_S = Ss[-1]
            last_X = Xs[-1]
            last_I = Is[-1]
            while True:
                last_dB1 = np.random.normal(loc=0, scale=1, size=1)
                last_dB2 = np.random.normal(loc=0, scale=1, size=1)
                next_X = self.next_X(last_X=last_X, last_dB2=last_dB2)
                next_S = self.next_S(last_S=last_S, last_I=last_I, last_dB1=last_dB1)
                next_I = self.next_I(last_I=last_I, last_S=last_S, last_X=last_X, last_dB1=last_dB1, last_dB2=last_dB2)
                if next_X < 0 and (0 <= next_S <= 1) and (0 <= next_I <= 1) and (next_S + next_I <= 1) and next_X>-1:
                    Ss.append(next_S)
                    Is.append(next_I)
                    Xs.append(next_X)
                    dB1.append(last_dB1)
                    dB2.append(last_dB2)
                    break
        return Ss, Is, Xs, dB1, dB2

    def get_data(self):
        Ss_trials = []
        Is_trials = []
        Xs_trials = []
        dB1_trials = []
        dB2_trials = []
        for idx in range(self.n_trials):
            Ss, Is, Xs, dB1, dB2 = self.get_data_one_trial
            Ss_trials.append(Ss)
            Is_trials.append(Is)
            Xs_trials.append(Xs)
            dB1_trials.append(dB1)
            dB2_trials.append(dB2)
        self.Ss_trials = np.array(Ss_trials)
        self.Is_trials = np.array(Is_trials)
        self.Xs_trials = np.array(Xs_trials)
        self.dB1_trials = dB1_trials
        self.dB2_trials = dB2_trials


# class DataModerateOU:
#     def __init__(self, I0, X0, S0, n_steps: int = 20, n_trials: int = 10000):
#         self.X0 = X0
#         self.I0 = I0
#         self.S0 = S0
#         self.n_trials = n_trials
#         self.n_steps = n_steps
#         self.d_B1 = None
#         self.d_B2 = None
#         self.Xs = None
#         self.Is = None
#         self.Ss = None
#         self.get_d_B1()
#         self.get_d_B2()
#         self.get_Xs()
#         self.get_Ss_Is()
#
#     def get_d_B1(self):
#         self.d_B1 = np.random.normal(loc=0, scale=1, size=(self.n_trials, self.n_steps))
#
#     def get_d_B2(self):
#         self.d_B2 = np.random.normal(loc=0, scale=1, size=(self.n_trials, self.n_steps))
#
#     def get_Xs_per_trial(self, trial_idx):
#         Xs_ = [self.X0]
#         d_B2 = self.d_B2[trial_idx]
#         for i in range(1, self.n_steps):
#             next_X = Xs_[-1] + conf.lambda_x * (conf.X_bar - Xs_[-1]) * conf.dt - conf.sigma_x * d_B2[i - 1]
#             if next_X >= 0:
#                 next_X = -0.001
#             Xs_.append(next_X)
#         return Xs_
#
#     def get_Xs(self):
#         self.Xs = np.array([self.get_Xs_per_trial(trial_idx=idx) for idx in range(self.n_trials)])
#
#     def get_Ss_Is_per_trial(self, trial_idx):
#         d_B1 = self.d_B1[trial_idx]
#         d_B2 = self.d_B2[trial_idx]
#         Xs_ = self.Xs[trial_idx]
#         Is_ = [self.I0]
#         Ss_ = [self.S0]
#         for i in range(1, self.n_steps):
#             next_S = Ss_[-1] - conf.beta * Ss_[-1] * Is_[-1] * conf.dt + conf.sigma_s * math.sqrt(Ss_[-1] * Is_[-1]) * \
#                      d_B1[i - 1]
#             next_I = Is_[-1] + (conf.beta * Ss_[-1] - conf.mu + conf.alpha_fix * conf.sigma * Xs_[i - 1]) * Is_[
#                 -1] * conf.dt + conf.alpha_fix * Is_[-1] * conf.sigma * d_B2[i - 1] - conf.sigma_s * math.sqrt(
#                 Ss_[-1] * Is_[-1]) * d_B1[i - 1]
#             if next_I <= 0:
#                 next_I = 0.001
#             if next_S <= 0:
#                 next_S = 0.001
#             if next_I >1:
#                 next_I = 1
#             if next_S >1:
#                 next_S = 1
#             Ss_.append(next_S)
#             Is_.append(next_I)
#         return Ss_, Is_
#
#     def get_Ss_Is(self):
#         Ss = []
#         Is = []
#         for idx in range(self.n_trials):
#             Ss_, Is_ = self.get_Ss_Is_per_trial(trial_idx=idx)
#             Ss.append(Ss_)
#             Is.append(Is_)
#         self.Ss = np.array(Ss)
#         self.Is = np.array(Is)


class DataModerateConst(DataModerateOU):
    def __init__(self, I0, S0, n_steps: int = 20, n_trials: int = 10000):
        super().__init__(I0=I0, X0=conf.X_bar, S0=S0, n_steps=n_steps, n_trials=n_trials)
        self.d_B1_trials = None
        self.d_B2_trials = None
        self.Xs_trials = None
        self.Is_trials = None
        self.Ss_trials = None
        self.get_data()

    def next_X(self, last_X, last_dB2):
        return conf.X_bar
