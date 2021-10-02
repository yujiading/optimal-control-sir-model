from functools import partial, reduce
from multiprocessing import Pool

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

    def parallel_helper(self, idx, data, gamma):
        try:
            conf.Xs = data.Xs_trials[idx]
        except:
            pass
        try:
            conf.Ss = data.Ss_trials[idx]
        except:
            pass
        conf.Is = data.Is_trials[idx]
        conf.length = len(conf.Is)
        conf.dIs = conf.Is[1:] - conf.Is[:-1]
        conf.dSs = conf.Ss[1:] - conf.Ss[:-1]
        conf.dXs = conf.Xs[1:] - conf.Xs[:-1]
        I, Utility = self.get_I_star_utility_dict(gamma=gamma)
        return I, Utility

    def get_I_star_utility_dict_simulation(self, gamma):
        data_simulation = self.simulation_dict[self.model]
        data = data_simulation(I0=self.I0, X0=self.X0, S0=self.S0, n_steps=self.n_steps, n_trials=self.n_trials)
        # data = DataModerateOU(I0=I0, X0=X0, S0=S0, n_steps=n_steps, n_trials=n_trials)
        paralell = True
        if paralell:
            trials = list(range(self.n_trials))
            with Pool(self.cpu) as p:
                ret = list(tqdm(p.imap(partial(self.parallel_helper,
                                               data=data, gamma=gamma), trials), total=self.n_trials))
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
                I, Utility = self.get_I_star_utility_dict_simulation(gamma=gamma)
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

# class Driver:
#     @staticmethod
#     def utility(I, gamma):
#         return -I ** (1 - gamma) / (1 - gamma)
#
#     @staticmethod
#     def get_I_star_I_full_low_const(gamma):
#         cla_alpha = AlphaStarLowConst(gamma=gamma)
#         alpha_star = cla_alpha.alpha_star
#         cla_I = IStarLowConst(alpha_star=alpha_star)
#         I_star_low_const = cla_I.I_star
#         cla_I_full = IStarLowConst(alpha_star=1)
#         I_full_low_const = cla_I_full.I_star
#         return I_star_low_const, I_full_low_const
#
#     @staticmethod
#     def get_I_star_I_full_low_OU(gamma, T):
#         cla_alpha = AlphaStarLowOU(gamma=gamma, T=T)
#         alpha_star = cla_alpha.alpha_star
#         # if conf.is_simulation:
#         #     alpha_star[alpha_star < 0] = 0
#         #     alpha_star[alpha_star > 1] = 1
#         # alpha_star = np.array(len(alpha_star)*[1.25])
#         cla_I = IStarLowOU(alpha_star=alpha_star)
#         I_star_low_OU = cla_I.I_star
#         cla_I_full = IStarLowOU(alpha_star=1)
#         I_full_low_OU = cla_I_full.I_star
#         return I_star_low_OU, I_full_low_OU
#
#     @staticmethod
#     def get_I_star_I_full_moderate_OU(gamma, T):
#         cla_alpha = AlphaStarModerateOU(gamma=gamma, T=T)
#         alpha_star = cla_alpha.alpha_star
#         if conf.is_simulation:
#             alpha_star[alpha_star < 0] = 0
#         cla_I = IStarModerateOU(alpha_star=alpha_star)
#         I_star_mod_OU = cla_I.I_star
#         cla_I_full = IStarModerateOU(alpha_star=1)
#         I_full_mod_OU = cla_I_full.I_star
#         return I_star_mod_OU, I_full_mod_OU
#
#     @staticmethod
#     def get_I_star_I_full_moderate_const(gamma, T):
#         cla_alpha = AlphaStarModerateConst(gamma=gamma, T=T)
#         alpha_star = cla_alpha.alpha_star
#         # if conf.is_simulation:
#         #     alpha_star[alpha_star < 0] = 0
#         cla_I = IStarModerateConst(alpha_star=alpha_star)
#         I_star_mod_const = cla_I.I_star
#         cla_I_full = IStarModerateConst(alpha_star=1)
#         I_full_mod_const = cla_I_full.I_star
#         return I_star_mod_const, I_full_mod_const
#
#     @staticmethod
#     def I_star_low_infection(is_plot_ultility: bool, T):
#         gammas = [-1, -2, -3, -4, -5]
#         styles = ['o-.', '*:', '<-.', '>-.', '^-.', '-', '--']
#         fig, axes = plt.subplots(nrows=len(gammas), ncols=2)
#
#         for i in range(len(gammas)):
#             gamma = gammas[i]
#             I = pd.DataFrame()
#
#             I_star_low_const, I_full_low_const = Driver.get_I_star_I_full_low_const(gamma=gamma)
#             I_star_low_OU, I_full_low_OU = Driver.get_I_star_I_full_low_OU(gamma=gamma, T=T)
#             I_no = conf.Is
#
#             if is_plot_ultility:
#                 I_star_low_const = Driver.utility(I=I_star_low_const, gamma=gamma)
#                 I_full_low_const = Driver.utility(I=I_full_low_const, gamma=gamma)
#                 I_star_low_OU = Driver.utility(I=I_star_low_OU, gamma=gamma)
#                 I_full_low_OU = Driver.utility(I=I_full_low_OU, gamma=gamma)
#                 I_no = Driver.utility(I=I_no, gamma=gamma)
#             I['Optimal Control with Constant Treatment'] = I_star_low_const
#             I['Optimal Control with OU Treatment'] = I_star_low_OU
#             I['Full Control with Constant Treatment'] = I_full_low_const
#             I['Full Control with OU Treatment'] = I_full_low_OU
#             I['No Control'] = I_no
#
#             I.plot(ax=axes[i, 0], style=styles, legend=False)
#             axes[i, 0].set_ylabel('$\\gamma=$' + str(gamma))
#             I.plot(ax=axes[i, 1], y=[
#                 'Optimal Control with Constant Treatment',
#                 'Optimal Control with OU Treatment',
#                 'Full Control with Constant Treatment',
#                 'Full Control with OU Treatment'],
#                    style=styles, legend=False)
#
#             yfmt = ScalarFormatterForceFormat()
#             yfmt.set_powerlimits((0, 0))
#             axes[i, 0].yaxis.set_major_formatter(yfmt)
#             yfmt = ScalarFormatterForceFormat()
#             yfmt.set_powerlimits((0, 0))
#             axes[i, 1].yaxis.set_major_formatter(yfmt)
#
#         handles, labels = axes[0, 0].get_legend_handles_labels()
#
#         # Format plot
#         fig.set_size_inches(8, 10.5)
#         fig.subplots_adjust(left=0.1, bottom=0.2, right=0.95, top=0.9, wspace=0.3, hspace=0.4)
#         fig.legend(handles, labels, bbox_to_anchor=(0.5, 0.025), loc='lower center')
#
#         if is_plot_ultility:
#             plt.suptitle('Utility of Low Infection Regime: $-\\frac{I(t)^{1-\\gamma}}{1-\\gamma}$', x=0.5)
#         else:
#             plt.suptitle('Infection of Low Infection Regime: $I$', x=0.5)
#         # fig.savefig(conf.cur_dir_path/'../plot/samplefigure', bbox_inches='tight')
#         plt.show()
#
#     @staticmethod
#     def I_star_moderate_infection(is_plot_ultility: bool, T, start_index):
#         gammas = [-1, -2, -3, -4, -5]
#         # gammas = [-11, -12, -13, -14, -15]
#         # gammas = [-.1, -.2, -.3, -.4, -.5]
#         styles = ['o-.', '*:', '<-.', '>-.', '^-.', '-', '--']
#         fig, axes = plt.subplots(nrows=len(gammas), ncols=2)
#
#         for i in range(len(gammas)):
#             gamma = gammas[i]
#             I = pd.DataFrame()
#
#             I_star_moderate_OU, I_full_moderate_OU = Driver.get_I_star_I_full_moderate_OU(gamma=gamma, T=T)
#             I_star_moderate_const, I_full_moderate_const = Driver.get_I_star_I_full_moderate_const(gamma=gamma, T=T)
#             I_no = conf.Is
#
#             if is_plot_ultility:
#                 I_star_moderate_OU = Driver.utility(I=I_star_moderate_OU, gamma=gamma)
#                 I_full_moderate_OU = Driver.utility(I=I_full_moderate_OU, gamma=gamma)
#                 I_star_moderate_const = Driver.utility(I=I_star_moderate_const, gamma=gamma)
#                 I_full_moderate_const = Driver.utility(I=I_full_moderate_const, gamma=gamma)
#                 I_no = Driver.utility(I=I_no, gamma=gamma)
#
#             I['Optimal Control with Constant Treatment'] = I_star_moderate_const
#             I['Optimal Control with OU Treatment'] = I_star_moderate_OU
#             I['Full Control with Constant Treatment'] = I_full_moderate_const
#             I['Full Control with OU Treatment'] = I_full_moderate_OU
#             I['No Control'] = I_no
#
#             I.index += start_index
#             I.plot(ax=axes[i, 0], style=styles, legend=False)
#             axes[i, 0].set_ylabel('$\\gamma=$' + str(gamma))
#             I.plot(ax=axes[i, 1], y=[
#                 'Optimal Control with Constant Treatment',
#                 'Optimal Control with OU Treatment',
#                 'Full Control with Constant Treatment',
#                 'Full Control with OU Treatment'],
#                    style=styles, legend=False)
#
#             yfmt = ScalarFormatterForceFormat()
#             yfmt.set_powerlimits((0, 0))
#             axes[i, 0].yaxis.set_major_formatter(yfmt)
#             yfmt = ScalarFormatterForceFormat()
#             yfmt.set_powerlimits((0, 0))
#             axes[i, 1].yaxis.set_major_formatter(yfmt)
#
#         handles, labels = axes[0, 0].get_legend_handles_labels()
#
#         # Format plot
#         fig.set_size_inches(8, 10.5)
#         fig.subplots_adjust(left=0.1, bottom=0.2, right=0.95, top=0.9, wspace=0.3, hspace=0.4)
#         fig.legend(handles, labels, bbox_to_anchor=(0.5, 0.025), loc='lower center')
#
#         if is_plot_ultility:
#             plt.suptitle('Utility of Moderate Infection Regime: $-\\frac{I(t)^{1-\\gamma}}{1-\\gamma}$', x=0.5)
#         else:
#             plt.suptitle('Infection of Moderate Infection Regime: $I$', x=0.5)
#         # fig.savefig(conf.cur_dir_path/'../plot/samplefigure', bbox_inches='tight')
#         plt.show()
#
#     @staticmethod
#     def I_star_moderate_infection_expect_parallel_helper(idx, data_ou, gamma, T):
#         conf.Xs = data_ou.Xs_trials[idx]
#         conf.Ss = data_ou.Ss_trials[idx]
#         conf.Is = data_ou.Is_trials[idx]
#         conf.length = len(conf.Is)
#         conf.dIs = conf.Is[1:] - conf.Is[:-1]
#         conf.dSs = conf.Ss[1:] - conf.Ss[:-1]
#         conf.dXs = conf.Xs[1:] - conf.Xs[:-1]
#
#         I_star_moderate_OU_, I_full_moderate_OU_ = Driver.get_I_star_I_full_moderate_OU(gamma=gamma, T=T)
#
#         # conf.Xs = data_const.Xs_trials[idx]
#         # conf.Ss = data_const.Ss_trials[idx]
#         # conf.Is = data_const.Is_trials[idx]
#         # conf.length = len(conf.Is)
#         # conf.dIs = conf.Is[1:] - conf.Is[:-1]
#         # conf.dSs = conf.Ss[1:] - conf.Ss[:-1]
#         # conf.dXs = conf.Xs[1:] - conf.Xs[:-1]
#         I_star_moderate_const_, I_full_moderate_const_ = Driver.get_I_star_I_full_moderate_const(gamma=gamma,
#                                                                                                  T=T)
#         return I_star_moderate_OU_, I_full_moderate_OU_, I_star_moderate_const_, I_full_moderate_const_
#
#     @staticmethod
#     def I_star_moderate_infection_expect(is_plot_ultility: bool, T):
#         # gammas = [-1, -2, -3, -4, -5]
#         # gammas = [-6, -7, -8, -9, -10]
#         gammas = [-1, -2]
#
#         data_ou = DataModerateOU(I0=conf.eps, X0=conf.X0, S0=conf.S0, n_steps=conf.n_steps, n_trials=conf.n_trials)
#         # data_const = DataModerateConst(I0=conf.eps, S0=conf.S0, n_steps=conf.n_steps, n_trials=conf.n_trials)
#         II = []
#         II_ultility = []
#         for i in range(len(gammas)):
#             gamma = gammas[i]
#             I = pd.DataFrame()
#             I_ultility = pd.DataFrame()
#             paralell = True
#             if paralell:
#                 trials = list(range(conf.n_trials))
#                 with Pool(conf.cpu) as p:
#                     ret = list(tqdm(p.imap(partial(Driver.I_star_moderate_infection_expect_parallel_helper,
#                                                    data_ou=data_ou, gamma=gamma, T=T), trials), total=conf.n_trials))
#                 I_star_moderate_OU, I_full_moderate_OU, I_star_moderate_const, I_full_moderate_const = np.sum(ret,
#                                                                                                               axis=0) / conf.n_trials
#             else:
#                 I_star_moderate_OU = 0
#                 I_full_moderate_OU = 0
#                 I_star_moderate_const = 0
#                 I_full_moderate_const = 0
#                 for idx in tqdm(range(conf.n_trials)):
#                     conf.Xs = data_ou.Xs_trials[idx]
#                     conf.Ss = data_ou.Ss_trials[idx]
#                     conf.Is = data_ou.Is_trials[idx]
#                     conf.length = len(conf.Is)
#                     conf.dIs = conf.Is[1:] - conf.Is[:-1]
#                     conf.dSs = conf.Ss[1:] - conf.Ss[:-1]
#                     conf.dXs = conf.Xs[1:] - conf.Xs[:-1]
#
#                     I_star_moderate_OU_, I_full_moderate_OU_ = Driver.get_I_star_I_full_moderate_OU(gamma=gamma, T=T)
#                     I_star_moderate_OU += I_star_moderate_OU_
#                     I_full_moderate_OU += I_full_moderate_OU_
#
#                     # conf.Xs = data_const.Xs_trials[idx]
#                     # conf.Ss = data_const.Ss_trials[idx]
#                     # conf.Is = data_const.Is_trials[idx]
#                     # conf.length = len(conf.Is)
#                     # conf.dIs = conf.Is[1:] - conf.Is[:-1]
#                     # conf.dSs = conf.Ss[1:] - conf.Ss[:-1]
#                     # conf.dXs = conf.Xs[1:] - conf.Xs[:-1]
#                     I_star_moderate_const_, I_full_moderate_const_ = Driver.get_I_star_I_full_moderate_const(
#                         gamma=gamma,
#                         T=T)
#                     I_star_moderate_const += I_star_moderate_const_
#                     I_full_moderate_const += I_full_moderate_const_
#                 I_star_moderate_OU = I_star_moderate_OU / conf.n_trials
#                 I_full_moderate_OU = I_full_moderate_OU / conf.n_trials
#                 I_star_moderate_const = I_star_moderate_const / conf.n_trials
#                 I_full_moderate_const = I_full_moderate_const / conf.n_trials
#
#             I_no = np.sum(data_ou.Is_trials, axis=0) / conf.n_trials
#             I['Optimal Control with Constant Treatment'] = I_star_moderate_const
#             I['Optimal Control with OU Treatment'] = I_star_moderate_OU
#             I['Full Control with Constant Treatment'] = I_full_moderate_const
#             I['Full Control with OU Treatment'] = I_full_moderate_OU
#             I['No Control'] = I_no
#             II.append(I)
#             if is_plot_ultility:
#                 I_ultility_star_moderate_OU = Driver.utility(I=I_star_moderate_OU, gamma=gamma)
#                 I_ultility_full_moderate_OU = Driver.utility(I=I_full_moderate_OU, gamma=gamma)
#                 I_ultility_star_moderate_const = Driver.utility(I=I_star_moderate_const, gamma=gamma)
#                 I_ultility_full_moderate_const = Driver.utility(I=I_full_moderate_const, gamma=gamma)
#                 I_ultility_no = Driver.utility(I=I_no, gamma=gamma)
#                 I_ultility['Optimal Control with Constant Treatment'] = I_ultility_star_moderate_const
#                 I_ultility['Optimal Control with OU Treatment'] = I_ultility_star_moderate_OU
#                 I_ultility['Full Control with Constant Treatment'] = I_ultility_full_moderate_const
#                 I_ultility['Full Control with OU Treatment'] = I_ultility_full_moderate_OU
#                 I_ultility['No Control'] = I_ultility_no
#                 II_ultility.append(I_ultility)
#
#         styles = ['C0o-.', 'C1*:', 'C2<-.', 'C3>-.', 'C4^-.', 'C5-', 'C6--']
#         fig, axes = plt.subplots(nrows=len(gammas), ncols=2)
#         for i in range(len(gammas)):
#             gamma = gammas[i]
#             I = II[i]
#             I.plot(ax=axes[i, 0], style=styles, legend=False)
#             axes[i, 0].set_ylabel('$\\gamma=$' + str(gamma))
#             I.plot(ax=axes[i, 1], y=[
#                 'Optimal Control with Constant Treatment',
#                 'Optimal Control with OU Treatment',
#                 'Full Control with Constant Treatment',
#                 'Full Control with OU Treatment'],
#                    style=styles, legend=False)
#
#             yfmt = ScalarFormatterForceFormat()
#             yfmt.set_powerlimits((0, 0))
#             axes[i, 0].yaxis.set_major_formatter(yfmt)
#             yfmt = ScalarFormatterForceFormat()
#             yfmt.set_powerlimits((0, 0))
#             axes[i, 1].yaxis.set_major_formatter(yfmt)
#
#         handles, labels = axes[0, 0].get_legend_handles_labels()
#
#         # Format plot
#         fig.set_size_inches(8, 10.5)
#         fig.subplots_adjust(left=0.1, bottom=0.2, right=0.95, top=0.9, wspace=0.3, hspace=0.4)
#         fig.legend(handles, labels, bbox_to_anchor=(0.5, 0.025), loc='lower center')
#         plt.suptitle('Expected Infection of Moderate Infection Regime: $EI$', x=0.5)
#         plt.show()
#         if is_plot_ultility:
#             styles = ['o-.', '*:', '<-.', '>-.', '^-.', '-', '--']
#             fig, axes = plt.subplots(nrows=len(gammas), ncols=2)
#             for i in range(len(gammas)):
#                 gamma = gammas[i]
#                 I = II_ultility[i]
#                 I.plot(ax=axes[i, 0], style=styles, legend=False)
#                 axes[i, 0].set_ylabel('$\\gamma=$' + str(gamma))
#                 I.plot(ax=axes[i, 1], y=[
#                     'Optimal Control with Constant Treatment',
#                     'Optimal Control with OU Treatment',
#                     'Full Control with Constant Treatment',
#                     'Full Control with OU Treatment'],
#                        style=styles, legend=False)
#
#                 yfmt = ScalarFormatterForceFormat()
#                 yfmt.set_powerlimits((0, 0))
#                 axes[i, 0].yaxis.set_major_formatter(yfmt)
#                 yfmt = ScalarFormatterForceFormat()
#                 yfmt.set_powerlimits((0, 0))
#                 axes[i, 1].yaxis.set_major_formatter(yfmt)
#
#             handles, labels = axes[0, 0].get_legend_handles_labels()
#
#             # Format plot
#             fig.set_size_inches(8, 10.5)
#             fig.subplots_adjust(left=0.1, bottom=0.2, right=0.95, top=0.9, wspace=0.3, hspace=0.4)
#             fig.legend(handles, labels, bbox_to_anchor=(0.5, 0.025), loc='lower center')
#             plt.suptitle('Expected Utility of Moderate Infection Regime: $E-\\frac{I(t)^{1-\\gamma}}{1-\\gamma}$',
#                          x=0.5)
#             plt.show()
#
#     @staticmethod
#     def I_star_low_infection_expect_parallel_helper(idx, data_ou, data_const, gamma, T):
#         conf.Xs = data_ou.Xs_trials[idx]
#         conf.Is = data_ou.Is_trials[idx]
#         conf.length = len(conf.Is)
#         conf.dIs = conf.Is[1:] - conf.Is[:-1]
#         conf.dXs = conf.Xs[1:] - conf.Xs[:-1]
#
#         I_star_low_OU_, I_full_low_OU_ = Driver.get_I_star_I_full_low_OU(gamma=gamma, T=T)
#
#         conf.Xs = data_const.Xs_trials[idx]
#         conf.Is = data_const.Is_trials[idx]
#         conf.length = len(conf.Is)
#         conf.dIs = conf.Is[1:] - conf.Is[:-1]
#         conf.dXs = conf.Xs[1:] - conf.Xs[:-1]
#         I_star_low_const_, I_full_low_const_ = Driver.get_I_star_I_full_low_const(gamma=gamma)
#         return I_star_low_OU_, I_full_low_OU_, I_star_low_const_, I_full_low_const_
#
#     @staticmethod
#     def I_star_low_infection_expect(is_plot_ultility: bool, T):
#         gammas = [-1, -2, -3, -4, -5]
#         # gammas = [-6, -7, -8, -9, -10]
#         # gammas = [-3, -4]
#
#         # data_ou = DataModerateOU(I0=conf.eps, X0=conf.X0, S0=conf.S0,n_steps=conf.n_steps, n_trials=conf.n_trials)
#         data_ou = DataLowOU(I0=conf.eps, X0=None, n_steps=conf.n_steps, n_trials=conf.n_trials)
#         data_const = DataLowConst(I0=conf.eps, n_steps=conf.n_steps, n_trials=conf.n_trials)
#         II = []
#         II_ultility = []
#         for i in range(len(gammas)):
#             gamma = gammas[i]
#             I = pd.DataFrame()
#             I_ultility = pd.DataFrame()
#             paralell = True
#             if paralell:
#                 trials = list(range(conf.n_trials))
#                 with Pool(conf.cpu) as p:
#                     ret = list(tqdm(p.imap(partial(Driver.I_star_low_infection_expect_parallel_helper,
#                                                    data_ou=data_ou, data_const=data_const, gamma=gamma, T=T), trials),
#                                     total=conf.n_trials))
#                 I_star_low_OU, I_full_low_OU, I_star_low_const, I_full_low_const = np.sum(ret,
#                                                                                           axis=0) / conf.n_trials
#             else:
#                 I_star_low_OU = 0
#                 I_full_low_OU = 0
#                 I_star_low_const = 0
#                 I_full_low_const = 0
#                 for idx in tqdm(range(conf.n_trials)):
#                     conf.Xs = data_ou.Xs_trials[idx]
#                     conf.Is = data_ou.Is_trials[idx]
#                     conf.length = len(conf.Is)
#                     conf.dIs = conf.Is[1:] - conf.Is[:-1]
#                     conf.dXs = conf.Xs[1:] - conf.Xs[:-1]
#
#                     I_star_low_OU_, I_full_low_OU_ = Driver.get_I_star_I_full_low_OU(gamma=gamma, T=T)
#                     I_star_low_OU += I_star_low_OU_
#                     I_full_low_OU += I_full_low_OU_
#
#                     # conf.Xs = data_const.Xs_trials[idx]
#                     # conf.Ss = data_const.Ss_trials[idx]
#                     # conf.Is = data_const.Is_trials[idx]
#                     # conf.length = len(conf.Is)
#                     # conf.dIs = conf.Is[1:] - conf.Is[:-1]
#                     # conf.dSs = conf.Ss[1:] - conf.Ss[:-1]
#                     # conf.dXs = conf.Xs[1:] - conf.Xs[:-1]
#                     I_star_low_const_, I_full_low_const_ = Driver.get_I_star_I_full_low_const(gamma=gamma)
#                     I_star_low_const += I_star_low_const_
#                     I_full_low_const += I_full_low_const_
#                 I_star_low_OU = I_star_low_OU / conf.n_trials
#                 I_full_low_OU = I_full_low_OU / conf.n_trials
#                 I_star_low_const = I_star_low_const / conf.n_trials
#                 I_full_low_const = I_full_low_const / conf.n_trials
#
#             I_no = np.sum(data_const.Is_trials, axis=0) / conf.n_trials
#             I['Optimal Control with Constant Treatment'] = I_star_low_const
#             I['Optimal Control with OU Treatment'] = I_star_low_OU
#             I['Full Control with Constant Treatment'] = I_full_low_const
#             I['Full Control with OU Treatment'] = I_full_low_OU
#             I['No Control'] = I_no
#             II.append(I)
#             if is_plot_ultility:
#                 I_ultility_star_low_OU = Driver.utility(I=I_star_low_OU, gamma=gamma)
#                 I_ultility_full_low_OU = Driver.utility(I=I_full_low_OU, gamma=gamma)
#                 I_ultility_star_low_const = Driver.utility(I=I_star_low_const, gamma=gamma)
#                 I_ultility_full_low_const = Driver.utility(I=I_full_low_const, gamma=gamma)
#                 I_ultility_no = Driver.utility(I=I_no, gamma=gamma)
#                 I_ultility['Optimal Control with Constant Treatment'] = I_ultility_star_low_const
#                 I_ultility['Optimal Control with OU Treatment'] = I_ultility_star_low_OU
#                 I_ultility['Full Control with Constant Treatment'] = I_ultility_full_low_const
#                 I_ultility['Full Control with OU Treatment'] = I_ultility_full_low_OU
#                 I_ultility['No Control'] = I_ultility_no
#                 II_ultility.append(I_ultility)
#
#         styles = ['C0o-.', 'C1*:', 'C2<-.', 'C3>-.', 'C4^-.', 'C5-', 'C6--']
#         fig, axes = plt.subplots(nrows=len(gammas), ncols=2)
#         for i in range(len(gammas)):
#             gamma = gammas[i]
#             I = II[i]
#             I.plot(ax=axes[i, 0], style=styles, legend=False)
#             axes[i, 0].set_ylabel('$\\gamma=$' + str(gamma))
#             I.plot(ax=axes[i, 1], y=[
#                 'Optimal Control with Constant Treatment',
#                 'Optimal Control with OU Treatment',
#                 'Full Control with Constant Treatment',
#                 'Full Control with OU Treatment'],
#                    style=styles, legend=False)
#
#             yfmt = ScalarFormatterForceFormat()
#             yfmt.set_powerlimits((0, 0))
#             axes[i, 0].yaxis.set_major_formatter(yfmt)
#             yfmt = ScalarFormatterForceFormat()
#             yfmt.set_powerlimits((0, 0))
#             axes[i, 1].yaxis.set_major_formatter(yfmt)
#
#         handles, labels = axes[0, 0].get_legend_handles_labels()
#
#         # Format plot
#         fig.set_size_inches(8, 10.5)
#         fig.subplots_adjust(left=0.1, bottom=0.2, right=0.95, top=0.9, wspace=0.3, hspace=0.4)
#         fig.legend(handles, labels, bbox_to_anchor=(0.5, 0.025), loc='lower center')
#         plt.suptitle('Expected Infection of Low Infection Regime: $EI$', x=0.5)
#         plt.show()
#         if is_plot_ultility:
#             styles = ['o-.', '*:', '<-.', '>-.', '^-.', '-', '--']
#             fig, axes = plt.subplots(nrows=len(gammas), ncols=2)
#             for i in range(len(gammas)):
#                 gamma = gammas[i]
#                 I = II_ultility[i]
#                 I.plot(ax=axes[i, 0], style=styles, legend=False)
#                 axes[i, 0].set_ylabel('$\\gamma=$' + str(gamma))
#                 I.plot(ax=axes[i, 1], y=[
#                     'Optimal Control with Constant Treatment',
#                     'Optimal Control with OU Treatment',
#                     'Full Control with Constant Treatment',
#                     'Full Control with OU Treatment'],
#                        style=styles, legend=False)
#
#                 yfmt = ScalarFormatterForceFormat()
#                 yfmt.set_powerlimits((0, 0))
#                 axes[i, 0].yaxis.set_major_formatter(yfmt)
#                 yfmt = ScalarFormatterForceFormat()
#                 yfmt.set_powerlimits((0, 0))
#                 axes[i, 1].yaxis.set_major_formatter(yfmt)
#
#             handles, labels = axes[0, 0].get_legend_handles_labels()
#
#             # Format plot
#             fig.set_size_inches(8, 10.5)
#             fig.subplots_adjust(left=0.1, bottom=0.2, right=0.95, top=0.9, wspace=0.3, hspace=0.4)
#             fig.legend(handles, labels, bbox_to_anchor=(0.5, 0.025), loc='lower center')
#             plt.suptitle('Expected Utility of Low Infection Regime: $E-\\frac{I(t)^{1-\\gamma}}{1-\\gamma}$',
#                          x=0.5)
#             plt.show()
