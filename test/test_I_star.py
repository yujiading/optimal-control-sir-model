import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from library import conf
from library.I_star import IStarLowConst
from library.alpha_star import AlphaStarLowConst
from library.data_simulation import DataLowConst

def test_I_star_low_const_utility():
    gammas = [-1, -2, -3, -4, -5]
    styles = ['o-.', '*:', '<-.', '>-.', '^-.', '-', '--']
    fig, axes = plt.subplots(nrows=len(gammas), ncols=2)
    for i in range(len(gammas)):
        gamma = gammas[i]

        U = pd.DataFrame()

        cla_alpha = AlphaStarLowConst(gamma=gamma)
        alpha_star = cla_alpha.alpha_star
        cla_I = IStarLowConst(alpha_star=alpha_star)
        I_star = cla_I.I_star
        U_star = -I_star ** (1 - gamma) / (1 - gamma)
        U['Optimal Control'] = U_star

        cla_I_full = IStarLowConst(alpha_star=1)
        I_full = cla_I_full.I_star
        U_full = -I_full ** (1 - gamma) / (1 - gamma)
        U['Full Control'] = U_full

        U_no = -conf.Is ** (1 - gamma) / (1 - gamma)
        U['No Control'] = U_no

        U.plot(ax=axes[i, 0], style=styles, legend=False)
        axes[i, 0].set_ylabel('$\\gamma=$' + str(gamma))
        U.plot(ax=axes[i, 1], y=['Optimal Control', 'Full Control'], style=styles, legend=False)

        axes[i, 0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        axes[i, 1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='best')
    plt.suptitle('$-\\frac{I(t)^{1-\\gamma}}{1-\\gamma}$ of Low Infection with Constant Treatment', x=0.35)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def test_I_star_low_const():
    gammas = [-1, -2, -3, -4, -5]
    styles = ['o-.', '*:', '<-.', '>-.', '^-.', '-', '--']
    fig, axes = plt.subplots(nrows=len(gammas), ncols=2)
    for i in range(len(gammas)):
        gamma = gammas[i]

        I = pd.DataFrame()

        cla_alpha = AlphaStarLowConst(gamma=gamma)
        alpha_star = cla_alpha.alpha_star
        cla_I = IStarLowConst(alpha_star=alpha_star)
        I_star = cla_I.I_star
        I['Optimal Control'] = I_star

        cla_I_full = IStarLowConst(alpha_star=1)
        I_full = cla_I_full.I_star
        I['Full Control'] = I_full

        I['No Control'] = conf.Is

        I.plot(ax=axes[i, 0], style=styles, legend=False)
        axes[i, 0].set_ylabel('$\\gamma=$' + str(gamma))
        I.plot(ax=axes[i, 1], y=['Optimal Control', 'Full Control'], style=styles, legend=False)

        axes[i, 0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        axes[i, 1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='best')
    plt.suptitle('$I$ of Low Infection with Constant Treatment', x=0.35)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def test_I_star_data_simulation():
    np.random.seed(0)
    Istar1 = IStarLowConst(alpha_star=1).I_star
    print()
    print(Istar1)
    np.random.seed(0)
    cla = DataLowConst(I0=conf.eps, n_steps=conf.length,n_trials=1)
    Istar2 = cla.Is_trials
    print(Istar2)