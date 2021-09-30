from library.plots import Plots
from library import conf
from library.data_simulation import DataModerateOU


def driver_low_real_data():
    Plots.I_star_low_infection(is_plot_ultility=True, T=1)
    Plots.I_star_low_infection(is_plot_ultility=False, T=1)

def driver_mod_real_data(start_index):
    """
    run e.g.: driver_mod_real_data(start_index=16)
    """
    conf.Xs = conf.Xs[start_index:]
    conf.Ss = conf.Ss[start_index:]
    conf.Is = conf.Is[start_index:]
    conf.length = len(conf.Is)
    conf.dIs = conf.Is[1:] - conf.Is[:-1]
    conf.dSs = conf.Ss[1:] - conf.Ss[:-1]
    conf.dXs = conf.Xs[1:] - conf.Xs[:-1]

    Plots.I_star_moderate_infection(is_plot_ultility=False, T=1, start_index=start_index)
    Plots.I_star_moderate_infection(is_plot_ultility=True, T=1, start_index=start_index)


def driver_mod_simulation():
    Plots.I_star_moderate_infection_expect(is_plot_ultility=True, T=1)



# uncomment to run
# driver_low_real_data()
if __name__ == '__main__':
    driver_mod_simulation()

# driver_mod_real_data(start_index=0)