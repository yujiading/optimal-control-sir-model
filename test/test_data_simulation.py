from library.data_simulation import DataModerateOU, DataModerateConst
from library import conf


def test_data_moderate_ou():
    data = DataModerateOU(I0=conf.eps, X0=conf.X0, S0=0.9, n_steps=10, n_trials=5)

    print(data.Is_trials)
    print(data.Xs_trials)
    print(data.Ss_trials)



def test_data_moderate_const():
    data = DataModerateConst(I0=conf.eps,S0=conf.S0, n_steps=10, n_trials=5)

    print(data.Is_trials)
    print(data.Xs_trials)
    print(data.Ss_trials)

