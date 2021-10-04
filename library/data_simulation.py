import numpy as np

import library.models.model_params
import library.run_config
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
        ret = last_X + library.models.model_params.lambda_x * (
                library.models.model_params.X_bar - last_X) * library.models.model_params.dt - library.models.model_params.sigma_x * last_dB2
        return ret[0]

    def next_S(self, last_S, last_I, last_dB1):
        ret = last_S - library.models.model_params.beta * last_S * last_I * library.models.model_params.dt + library.models.model_params.sigma_s * math.sqrt(
            last_S * last_I) * last_dB1
        return ret[0]

    def next_I(self, last_I, last_S, last_X, last_dB1, last_dB2):
        ret = last_I + (
                library.models.model_params.beta * last_S - library.models.model_params.mu + library.run_config.alpha_fix * library.models.model_params.sigma * last_X) * last_I * library.models.model_params.dt \
              + library.run_config.alpha_fix * last_I * library.models.model_params.sigma * last_dB2 - library.models.model_params.sigma_s * math.sqrt(
            last_S * last_I) * last_dB1
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
                last_dB1 = np.random.normal(loc=0, scale=math.sqrt(library.models.model_params.dt), size=1)
                last_dB2 = np.random.normal(loc=0, scale=math.sqrt(library.models.model_params.dt), size=1)
                next_X = self.next_X(last_X=last_X, last_dB2=last_dB2)
                next_S = self.next_S(last_S=last_S, last_I=last_I, last_dB1=last_dB1)
                next_I = self.next_I(last_I=last_I, last_S=last_S, last_X=last_X, last_dB1=last_dB1, last_dB2=last_dB2)
                if next_X < 0 and (0 <= next_S <= 1) and (0 <= next_I <= 1) and (next_S + next_I <= 1) and next_X > -1:
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


class DataModerateConst(DataModerateOU):
    def __init__(self, I0, S0, X0=None, n_steps: int = 20, n_trials: int = 10000):
        super().__init__(I0=I0, X0=library.models.model_params.X_bar, S0=S0, n_steps=n_steps, n_trials=n_trials)

    def next_X(self, last_X, last_dB2):
        return library.models.model_params.X_bar


class DataLowOU:
    def __init__(self, I0, X0, S0=None, n_steps: int = 20, n_trials: int = 10000):
        self.X0 = X0
        self.I0 = I0
        # if self.I0 is None:
        #     self.I0 = np.random.uniform(library.models.model_params.eps, 0.1)
        # if self.X0 is None:
        #     self.X0 = np.random.uniform(-0.5, 0.5)
        self.n_trials = n_trials
        self.n_steps = n_steps
        self.d_B2_trials = None
        self.Xs_trials = None
        self.Is_trials = None
        self.get_data()

    def next_X(self, last_X, last_dB2):
        ret = last_X + library.models.model_params.lambda_x * (
                library.models.model_params.X_bar - last_X) * library.models.model_params.dt - library.models.model_params.sigma_x * last_dB2
        return ret

    def next_I(self, last_I, last_X, last_dB2):
        ret = last_I + (
                library.models.model_params.r + library.run_config.alpha_fix * library.models.model_params.sigma * last_X) * last_I * library.models.model_params.dt \
              + library.run_config.alpha_fix * last_I * library.models.model_params.sigma * last_dB2
        return ret

    @property
    def get_data_one_trial(self):
        Xs = [self.X0]
        Is = [self.I0]
        dB2 = []
        # np.random.seed(0)
        for i in range(1, self.n_steps):
            last_X = Xs[-1]
            last_I = Is[-1]
            while True:
                last_dB2 = np.random.normal(loc=0, scale=math.sqrt(library.models.model_params.dt), size=1)[0]
                # last_dB2 = np.random.normal(loc=0, scale=conf.dt, size=1)[0]
                next_X = self.next_X(last_X=last_X, last_dB2=last_dB2)
                next_I = self.next_I(last_I=last_I, last_X=last_X, last_dB2=last_dB2)
                # if next_X < 0 and (0 <= next_I <= 1) and next_X > -1:
                if next_I <= 0:
                    next_I = 0
                Is.append(next_I)
                Xs.append(next_X)
                dB2.append(last_dB2)
                break
        # print(f'dB2 {dB2}')
        return Is, Xs, dB2

    def get_data(self):
        Is_trials = []
        Xs_trials = []
        dB2_trials = []
        for idx in range(self.n_trials):
            Is, Xs, dB2 = self.get_data_one_trial
            Is_trials.append(Is)
            Xs_trials.append(Xs)
            dB2_trials.append(dB2)
        self.Is_trials = np.array(Is_trials)
        self.Xs_trials = np.array(Xs_trials)
        self.dB2_trials = dB2_trials


class DataLowConst(DataLowOU):
    def __init__(self, I0, X0=None, S0=None, n_steps: int = 20, n_trials: int = 10000):
        super().__init__(I0=I0, X0=library.models.model_params.X_bar, n_steps=n_steps, n_trials=n_trials)

    def next_X(self, last_X, last_dB2):
        return library.models.model_params.X_bar
