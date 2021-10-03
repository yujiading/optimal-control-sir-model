import library.models.model_params
from library.data_simulation import DataModerateOU, DataModerateConst, DataLowConst, DataLowOU
from library import conf


def test_data_moderate_ou():
    data = DataModerateOU(I0=library.models.model_params.eps, X0=library.models.model_params.X0, S0=0.9, n_steps=10, n_trials=5)

    print(data.Is_trials)
    print(data.Xs_trials)
    print(data.Ss_trials)


def test_data_moderate_const():
    data = DataModerateConst(I0=library.models.model_params.eps, S0=library.models.model_params.S0, n_steps=10, n_trials=5)

    print(data.Is_trials)
    print(data.Xs_trials)
    print(data.Ss_trials)


def test_data_low_const():
    data = DataLowConst(I0=library.models.model_params.eps, n_steps=10, n_trials=5)
    print(data.Is_trials)
    print(data.Xs_trials)


def test_data_low_ou():
    data = DataLowOU(I0=library.models.model_params.eps, X0=library.models.model_params.X0, n_steps=20, n_trials=5)
    print(data.Is_trials)
    print(data.Xs_trials)
