from functools import partial, reduce
from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import ScalarFormatter
from tqdm import tqdm

from library import conf
from library.I_star import IStarLowConst, IStarLowOU, IStarModerateOU, IStarModerateConst
from library.alpha_star import AlphaStarLowConst, AlphaStarLowOU, AlphaStarModerateOU, AlphaStarModerateConst
from library.data_simulation import DataModerateOU, DataLowOU, DataLowConst, DataModerateConst


class Driver:
    def __init__(self, model, gammas, T, is_simulation: bool = True, X0=None, I0=None, S0=None, cpu: int = 8,
                 n_trials: int = 1000, n_steps: int = 20, is_include_optimal_control: bool = True,
                 is_include_full_control: bool = True,
                 is_include_no_control: bool = True):
        """
        requires X0, I0, S0, cpu, n_trials, n_steps if is_simulation is True
        """
        self.model = model
        self.gammas = gammas
        self.T = T
        self.is_simulation = is_simulation
        self.X0 = X0
        self.I0 = I0
        self.S0 = S0
        self.cpu = cpu
        self.n_trials = n_trials
        self.n_steps = n_steps
        self.is_include_optimal_control = is_include_optimal_control
        self.is_include_full_control = is_include_full_control
        self.is_include_no_control = is_include_no_control

        self.model_dict = {"LowConst": {"AlphaStar": AlphaStarLowConst,
                                        "IStar": IStarLowConst},
                           "LowOU": {"AlphaStar": AlphaStarLowOU,
                                     "IStar": IStarLowOU},
                           "ModerateConst": {"AlphaStar": AlphaStarModerateConst,
                                             "IStar": IStarModerateConst},
                           "ModerateOU": {"AlphaStar": AlphaStarModerateOU,
                                          "IStar": IStarModerateOU}
                           }

        self.simulation_dict = {"LowConst": DataLowConst,
                                "LowOU": DataLowOU,
                                "ModerateConst": DataModerateConst,
                                "ModerateOU": DataModerateOU}

        self.treatment_type_dict = {"LowConst": "Constant",
                                    "LowOU": "OU",
                                    "ModerateConst": "Constant",
                                    "ModerateOU": "OU"}
        self.infection_type_dict = {"LowConst": "Low",
                                    "LowOU": "Low",
                                    "ModerateConst": "Moderate",
                                    "ModerateOU": "Moderate"}

    @staticmethod
    def utility(I, gamma):
        return -I ** (1 - gamma) / (1 - gamma)

    def get_alpha_star(self, gamma):
        cla_alpha = self.model_dict[self.model]["AlphaStar"](gamma=gamma, T=self.T)
        alpha_star = cla_alpha.alpha_star
        # if conf.is_simulation:
        #     alpha_star[alpha_star < 0] = 0
        #     alpha_star[alpha_star > 1] = 1
        # alpha_star = np.array(len(alpha_star)*[1.25])
        return alpha_star

    def get_I_star(self, alpha_star):
        cla_I = self.model_dict[self.model]["IStar"](alpha_star=alpha_star)
        I_star = cla_I.I_star
        return I_star

    def get_I_star_utility_dict(self, gamma):
        I = pd.DataFrame()
        Utility = pd.DataFrame()
        if self.is_include_optimal_control:
            alpha_star = self.get_alpha_star(gamma=gamma)
            I_optimal = self.get_I_star(alpha_star=alpha_star)
            I[f'Optimal Control'] = I_optimal
            Utility[f'Optimal Control'] = Driver.utility(I=I_optimal, gamma=gamma)
        if self.is_include_full_control:
            I_full = self.get_I_star(alpha_star=1)
            I[f'Full Control'] = I_full
            Utility[f'Full Control'] = Driver.utility(I=I_full, gamma=gamma)
        if self.is_include_no_control:
            I_no = conf.Is
            I['No Control'] = I_no
            Utility[f'No Control'] = Driver.utility(I=I_no, gamma=gamma)
        return I, Utility

    def parallel_helper(self, data_elm, gamma):
        """
        data: [(Xs_trials, Ss_trials, Is_trials), ...]
        """
        conf.Xs, conf.Ss, conf.Is = data_elm
        conf.length = len(conf.Is)
        conf.dIs = conf.Is[1:] - conf.Is[:-1]
        conf.dSs = conf.Ss[1:] - conf.Ss[:-1]
        conf.dXs = conf.Xs[1:] - conf.Xs[:-1]
        I, Utility = self.get_I_star_utility_dict(gamma=gamma)
        return I, Utility

    def get_I_star_utility_dict_simulation(self, gamma, data):
        """
        data: [(data.Xs_trials, data.Ss_trials , data.Is_trials), ... ]
        """
        paralell = True
        if paralell:
            trials = list(range(self.n_trials))
            with Pool(self.cpu) as p:
                ret = list(tqdm(p.imap(partial(self.parallel_helper, gamma=gamma), data), total=self.n_trials))
            I_list = [item[0] for item in ret]
            Utility_list = [item[1] for item in ret]
            I = reduce(lambda a, b: a.add(b, fill_value=0), I_list)
            I = I / self.n_trials
            Utility = reduce(lambda a, b: a.add(b, fill_value=0), Utility_list)
            Utility = Utility / self.n_trials
        else:
            pass
        return I, Utility

    def plot(self):
        infection_type = self.infection_type_dict[self.model]
        treatment_type = self.treatment_type_dict[self.model]
        data_simulation = self.simulation_dict[self.model]
        data = data_simulation(I0=self.I0, X0=self.X0, S0=self.S0, n_steps=self.n_steps, n_trials=self.n_trials)
        if not hasattr(data, 'Ss_trials'):
            data.Ss_trials = np.zeros((self.n_trials, self.n_steps))
        if not hasattr(data, 'Is_trials'):
            data.Is_trials = np.zeros((self.n_trials, self.n_steps))
        dataset = list(zip(list(data.Xs_trials), list(data.Ss_trials), list(data.Is_trials)))
        # data = DataModerateOU(I0=I0, X0=X0, S0=S0, n_steps=n_steps, n_trials=n_trials)
        styles = ['C0o-.', 'C1*:', 'C2<-.', 'C3>-.', 'C4^-.', 'C5-', 'C6--']
        fig, axes = plt.subplots(nrows=len(self.gammas), ncols=2)
        if self.is_simulation:
            subtitle_I = f"Expected Infection: $EI(t)$"
            subtitle_utility = "Expected Utility: $E-\\frac{I(t)^{1-\\gamma}}{1-\\gamma}$"
        else:
            subtitle_I = f"Infection: $I(t)$"
            subtitle_utility = "Utility: $-\\frac{I(t)^{1-\\gamma}}{1-\\gamma}$"
        for i in range(len(self.gammas)):
            gamma = self.gammas[i]
            if self.is_simulation:
                I, Utility = self.get_I_star_utility_dict_simulation(gamma=gamma, data=dataset)
            else:
                I, Utility = self.get_I_star_utility_dict(gamma=gamma)
            if i > 0:
                subtitle_I = None
                subtitle_utility = None
            I.plot(ax=axes[i, 0], style=styles, legend=False, title=subtitle_I, sharex=True)
            axes[i, 0].set_ylabel('$\\gamma=$' + str(gamma))
            Utility.plot(ax=axes[i, 1], style=styles, legend=False, title=subtitle_utility)
            yfmt = ScalarFormatterForceFormat()
            yfmt.set_powerlimits((0, 0))
            axes[i, 0].yaxis.set_major_formatter(yfmt)
            yfmt = ScalarFormatterForceFormat()
            yfmt.set_powerlimits((0, 0))
            axes[i, 1].yaxis.set_major_formatter(yfmt)

        handles, labels = axes[0, 0].get_legend_handles_labels()

        # Format plot
        fig.set_size_inches(8, 10.5)
        fig.subplots_adjust(left=0.1, bottom=0.2, right=0.95, top=0.9, wspace=0.3, hspace=0.4)
        fig.legend(handles, labels, bbox_to_anchor=(0.5, 0.025), loc='lower center')
        plt.suptitle(f'{infection_type} Infection Regime with {treatment_type} Treatment', x=0.5)
        plt.show()

        # Format plot
        fig.set_size_inches(8, 10.5)
        fig.subplots_adjust(left=0.1, bottom=0.2, right=0.95, top=0.9, wspace=0.3, hspace=0.4)
        plt.show()


class ScalarFormatterForceFormat(ScalarFormatter):
    pass
    # def _set_format(self):  # Override function that finds format to use.
    #     self.format = "%1.1f"  # Give format here
