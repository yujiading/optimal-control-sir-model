import pathlib

import numpy as np
import pandas as pd

cur_dir_path = pathlib.Path(__file__).parent

data = pd.read_csv(cur_dir_path / 'data.csv')
Xs = np.array(data['X(t)'])
ts = np.array(data['t'])
Is = np.array(data['I(t)'])
Ss = np.array(data['S(t)'])

dIs = Is[1:] - Is[:-1]
dSs = Ss[1:] - Ss[:-1]
length = len(Is)
alpha_fix = 0.25

# dataset 2
sigma_k = -1.164695  # 5.876383  # Volatility of changes in the recovery rate
sigma = 0.4418  # Volatility of the measurement of today’s recovery rate
lambda_k = 0.7692  # Speed of mean-reversion of the recovery rate
K0 = 0.2559  # Recovery rate/ no treatment
mu0 = 0.0575  # Death rate/no treatment
mu1 = 0.0575  # Death rate
k1_bar = 0.4612  # 4 # Long run value of recovery rate

# dataset 22
# sigma_k = -1.164695  # 5.876383  # Volatility of changes in the recovery rate
# sigma = 0.4418  # Volatility of the measurement of today’s recovery rate
# lambda_k = 0.7692  # Speed of mean-reversion of the recovery rate
# K0 = 0.2559  # Recovery rate/ no treatment
# mu0 = 0.0575  # Death rate/no treatment
# mu1 = 0.0575  # Death rate
# k1_bar = 6  # 4 # Long run value of recovery rate


sigma_x = sigma_k / sigma  # needs to be negative
lambda_x = lambda_k
mu = K0 + mu0
X_bar = (mu - mu1 - k1_bar) / sigma  # long run impact of the treatment risk

# new parameters
beta = 0.025  # constant transmission rate  1.5-3.5
# https://www.statista.com/statistics/1103196/worldwide-infection-rate-of-major-virus-outbreaks/
sigma_s = 1.5  # Volatility of the measurement of today’s susceptible rate
dt = 0.001
eps = 0.01
