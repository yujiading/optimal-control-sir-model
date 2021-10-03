import pathlib

import numpy as np
import pandas as pd

cur_dir_path = pathlib.Path(__file__).parent

real_world_data = pd.read_csv(cur_dir_path / 'data.csv')
# Xs = np.array(data['X(t)'])
# ts = np.array(data['t'])
# Is = np.array(data['I(t)'])
# Ss = np.array(data['S(t)'])

# dIs = Is[1:] - Is[:-1]
# dSs = Ss[1:] - Ss[:-1]
# dXs = Xs[1:] - Xs[:-1]
# length = len(Is)
