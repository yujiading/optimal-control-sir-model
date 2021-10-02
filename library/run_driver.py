from library.driver import Driver
from library import conf
from library.data_simulation import DataModerateOU
import numpy as np


def run():
    """
    model takes values: 'LowConst', 'LowOU', 'ModerateConst', 'ModerateOU'
    """
    driver = Driver(model='LowConst', gammas=[-1], T=1, is_simulation=True, X0=conf.X0, I0=conf.eps, S0=conf.S0,
                    cpu=8, n_trials=1000, n_steps=conf.n_steps, is_include_optimal_control=True,
                    is_include_full_control=True, is_include_no_control=True)

    driver.plot()


# uncomment to run
# driver_low_real_data()
if __name__ == '__main__':
    # driver_mod_simulation()
    # np.random.seed(2)
    run()
    # driver_low_real_data()

# driver_mod_real_data(start_index=0)


# def driver_low_real_data():
#     Driver.I_star_low_infection(is_plot_ultility=True, T=1)
#     Driver.I_star_low_infection(is_plot_ultility=False, T=1)
#
# def driver_mod_real_data(start_index):
#     """
#     run e.g.: driver_mod_real_data(start_index=16)
#     """
#     conf.Xs = conf.Xs[start_index:]
#     conf.Ss = conf.Ss[start_index:]
#     conf.Is = conf.Is[start_index:]
#     conf.length = len(conf.Is)
#     conf.dIs = conf.Is[1:] - conf.Is[:-1]
#     conf.dSs = conf.Ss[1:] - conf.Ss[:-1]
#     conf.dXs = conf.Xs[1:] - conf.Xs[:-1]
#
#     Driver.I_star_moderate_infection(is_plot_ultility=False, T=1, start_index=start_index)
#     Driver.I_star_moderate_infection(is_plot_ultility=True, T=1, start_index=start_index)
#
#
# def driver_mod_simulation():
#     Driver.I_star_moderate_infection_expect(is_plot_ultility=True, T=1)
#
# def driver_low_simulation():
#     Driver.I_star_low_infection_expect(is_plot_ultility=True, T=0.1)
