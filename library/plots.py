import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import ScalarFormatter

from library import conf
from library.I_star import IStarLowConst, IStarLowOU, IStarModerateOU, IStarModerateConst
from library.alpha_star import AlphaStarLowConst, AlphaStarLowOU, AlphaStarModerateOU, AlphaStarModerateConst


class Plots:
    @staticmethod
    def utility(I, gamma):
        return -I ** (1 - gamma) / (1 - gamma)

    @staticmethod
    def get_I_star_I_full_low_const(gamma):
        cla_alpha = AlphaStarLowConst(gamma=gamma)
        alpha_star = cla_alpha.alpha_star
        cla_I = IStarLowConst(alpha_star=alpha_star)
        I_star_low_const = cla_I.I_star
        cla_I_full = IStarLowConst(alpha_star=1)
        I_full_low_const = cla_I_full.I_star
        return I_star_low_const, I_full_low_const

    @staticmethod
    def get_I_star_I_full_low_OU(gamma, T):
        cla_alpha = AlphaStarLowOU(gamma=gamma, T=T)
        alpha_star = cla_alpha.alpha_star
        cla_I = IStarLowOU(alpha_star=alpha_star)
        I_star_low_OU = cla_I.I_star
        cla_I_full = IStarLowOU(alpha_star=1)
        I_full_low_OU = cla_I_full.I_star
        return I_star_low_OU, I_full_low_OU

    @staticmethod
    def get_I_star_I_full_moderate_OU(gamma, T):
        cla_alpha = AlphaStarModerateOU(gamma=gamma, T=T)
        alpha_star = cla_alpha.alpha_star
        # alpha_star[alpha_star < 0] = 0
        cla_I = IStarModerateOU(alpha_star=alpha_star)
        I_star_mod_OU = cla_I.I_star
        cla_I_full = IStarModerateOU(alpha_star=1)
        I_full_mod_OU = cla_I_full.I_star
        return I_star_mod_OU, I_full_mod_OU

    @staticmethod
    def get_I_star_I_full_moderate_const(gamma, T):
        cla_alpha = AlphaStarModerateConst(gamma=gamma, T=T)
        alpha_star = cla_alpha.alpha_star
        # alpha_star[alpha_star < 0] = 0
        cla_I = IStarModerateConst(alpha_star=alpha_star)
        I_star_mod_const = cla_I.I_star
        cla_I_full = IStarModerateConst(alpha_star=1)
        I_full_mod_const = cla_I_full.I_star
        return I_star_mod_const, I_full_mod_const

    @staticmethod
    def I_star_low_infection(is_plot_ultility: bool, T):
        gammas = [-1, -2, -3, -4, -5]
        styles = ['o-.', '*:', '<-.', '>-.', '^-.', '-', '--']
        fig, axes = plt.subplots(nrows=len(gammas), ncols=2)

        for i in range(len(gammas)):
            gamma = gammas[i]
            I = pd.DataFrame()

            I_star_low_const, I_full_low_const = Plots.get_I_star_I_full_low_const(gamma=gamma)
            I_star_low_OU, I_full_low_OU = Plots.get_I_star_I_full_low_OU(gamma=gamma, T=T)
            I_no = conf.Is

            if is_plot_ultility:
                I_star_low_const = Plots.utility(I=I_star_low_const, gamma=gamma)
                I_full_low_const = Plots.utility(I=I_full_low_const, gamma=gamma)
                I_star_low_OU = Plots.utility(I=I_star_low_OU, gamma=gamma)
                I_full_low_OU = Plots.utility(I=I_full_low_OU, gamma=gamma)
                I_no = Plots.utility(I=I_no, gamma=gamma)
            I['Optimal Control with Constant Treatment'] = I_star_low_const
            I['Optimal Control with OU Treatment'] = I_star_low_OU
            I['Full Control with Constant Treatment'] = I_full_low_const
            I['Full Control with OU Treatment'] = I_full_low_OU
            I['No Control'] = I_no

            I.plot(ax=axes[i, 0], style=styles, legend=False)
            axes[i, 0].set_ylabel('$\\gamma=$' + str(gamma))
            I.plot(ax=axes[i, 1], y=[
                'Optimal Control with Constant Treatment',
                'Optimal Control with OU Treatment',
                'Full Control with Constant Treatment',
                'Full Control with OU Treatment'],
                   style=styles, legend=False)

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

        if is_plot_ultility:
            plt.suptitle('Utility of Low Infection Regime: $-\\frac{I(t)^{1-\\gamma}}{1-\\gamma}$', x=0.5)
        else:
            plt.suptitle('Infection of Low Infection Regime: $I$', x=0.5)
        # fig.savefig(conf.cur_dir_path/'../plot/samplefigure', bbox_inches='tight')
        plt.show()

    @staticmethod
    def I_star_moderate_infection(is_plot_ultility: bool, T, start_index):
        gammas = [-1, -2, -3, -4, -5]
        # gammas = [-11, -12, -13, -14, -15]
        styles = ['o-.', '*:', '<-.', '>-.', '^-.', '-', '--']
        fig, axes = plt.subplots(nrows=len(gammas), ncols=2)

        for i in range(len(gammas)):
            gamma = gammas[i]
            I = pd.DataFrame()

            I_star_moderate_OU, I_full_moderate_OU = Plots.get_I_star_I_full_moderate_OU(gamma=gamma, T=T)
            I_star_moderate_const, I_full_moderate_const = Plots.get_I_star_I_full_moderate_const(gamma=gamma, T=T)
            I_no = conf.Is

            if is_plot_ultility:
                I_star_moderate_OU = Plots.utility(I=I_star_moderate_OU, gamma=gamma)
                I_full_moderate_OU = Plots.utility(I=I_full_moderate_OU, gamma=gamma)
                I_star_moderate_const = Plots.utility(I=I_star_moderate_const, gamma=gamma)
                I_full_moderate_const = Plots.utility(I=I_full_moderate_const, gamma=gamma)
                I_no = Plots.utility(I=I_no, gamma=gamma)
            I['Optimal Control with OU Treatment'] = I_star_moderate_OU
            I['Optimal Control with Constant Treatment'] = I_star_moderate_const
            I['Full Control with OU Treatment'] = I_full_moderate_OU
            I['Full Control with Constant Treatment'] = I_full_moderate_const
            I['No Control'] = I_no

            I.index += start_index
            I.plot(ax=axes[i, 0], style=styles, legend=False)
            axes[i, 0].set_ylabel('$\\gamma=$' + str(gamma))
            I.plot(ax=axes[i, 1], y=[
                'Optimal Control with OU Treatment',
                'Optimal Control with Constant Treatment',
                'Full Control with OU Treatment',
                'Full Control with Constant Treatment'],
                   style=styles, legend=False)

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

        if is_plot_ultility:
            plt.suptitle('Utility of Moderate Infection Regime: $-\\frac{I(t)^{1-\\gamma}}{1-\\gamma}$', x=0.5)
        else:
            plt.suptitle('Infection of Moderate Infection Regime: $I$', x=0.5)
        # fig.savefig(conf.cur_dir_path/'../plot/samplefigure', bbox_inches='tight')
        plt.show()


class ScalarFormatterForceFormat(ScalarFormatter):
    def _set_format(self):  # Override function that finds format to use.
        self.format = "%1.1f"  # Give format here

################
# class Plots:
#     @staticmethod
#     def utility(I, gamma):
#         return -I ** (1 - gamma) / (1 - gamma)
#
#     @staticmethod
#     def get_I_star(model_name, gamma, T=None):
#         """
#         model name: low const, low OU, mod const, mod OU
#         """
#         I_star = None
#         I_full = None
#         if model_name == 'low const':
#             cla_alpha = AlphaStarLowConst(gamma=gamma)
#             alpha_star = cla_alpha.alpha_star
#             cla_I = IStarLowConst(alpha_star=alpha_star, eps=conf.eps)
#             I_star = cla_I.I_star
#             cla_I_full = IStarLowConst(alpha_star=1, eps=conf.eps)
#             I_full = cla_I_full.I_star
#         if model_name == 'low OU':
#             cla_alpha = AlphaStarLowOU(gamma=gamma, T=T)
#             alpha_star = cla_alpha.alpha_star
#             cla_I = IStarLowOU(alpha_star=alpha_star, eps=conf.eps, T=T, gamma=gamma)
#             I_star = cla_I.I_star
#             cla_I_full = IStarLowOU(alpha_star=1, eps=conf.eps, T=T, gamma=gamma)
#             I_full = cla_I_full.I_star
#         return I_star, I_full
#
#     @staticmethod
#     def I_star(model_name, is_plot_ultility: bool, T=None):
#         """
#         model name: low const, low OU, mod const, mod OU
#         """
#         gammas = [-1, -2, -3, -4, -5]
#         styles = ['o-.', '*:', '<-.', '>-.', '^-.', '-', '--']
#         fig, axes = plt.subplots(nrows=len(gammas), ncols=2)
#         for i in range(len(gammas)):
#             gamma = gammas[i]
#             I = pd.DataFrame()
#
#             I_star, I_full = Plots.get_I_star(model_name=model_name, gamma=gamma, T=T)
#             I_no = conf.Is
#
#             if is_plot_ultility:
#                 I_star = Plots.utility(I=I_star, gamma=gamma)
#                 I_full = Plots.utility(I=I_full, gamma=gamma)
#                 I_no = Plots.utility(I=I_no, gamma=gamma)
#             I['Optimal Control'] = I_star
#             I['Full Control'] = I_full
#             I['No Control'] = I_no
#
#             I.plot(ax=axes[i, 0], style=styles, legend=False)
#             axes[i, 0].set_ylabel('$\\gamma=$' + str(gamma))
#             I.plot(ax=axes[i, 1], y=['Optimal Control', 'Full Control'], style=styles, legend=False)
#
#             # axes[i, 0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
#             # axes[i, 1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
#             yfmt = ScalarFormatterForceFormat()
#             yfmt.set_powerlimits((0, 0))
#             axes[i, 0].yaxis.set_major_formatter(yfmt)
#             yfmt = ScalarFormatterForceFormat()
#             yfmt.set_powerlimits((0, 0))
#             axes[i, 1].yaxis.set_major_formatter(yfmt)
#
#         handles, labels = axes[0, 0].get_legend_handles_labels()
#         fig.legend(handles, labels, loc='upper right')
#         if is_plot_ultility:
#             plt.suptitle('$-\\frac{I(t)^{1-\\gamma}}{1-\\gamma}$ of Low Infection with Constant Treatment', x=0.35)
#         else:
#             plt.suptitle('$I$ of Low Infection with Constant Treatment', x=0.35)
#         fig.tight_layout(rect=[0, 0.03, 1, 0.95])
#         plt.show()
