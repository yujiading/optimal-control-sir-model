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
from models.model_mapper import ModelTypes, VariableNames, model_class_map
from models import model_params
from plot_generator import PlotGenerator
from run_config import RunConfig


class Driver:
    def __init__(self, run_config: RunConfig):
        """
        requires X0, I0, S0, cpu, n_trials, n_steps if is_simulation is True
        """
        self.run_config = run_config

        self.simulation_dict = {"LowConst": DataLowConst,
                                "LowOU": DataLowOU,
                                "ModerateConst": DataModerateConst,
                                "ModerateOU": DataModerateOU}

    @staticmethod
    def utility(I, gamma):
        return -I ** (1 - gamma) / (1 - gamma)

    def get_alpha_star(self, gamma, Xs, Ss, Is):
        model_class_alpha = model_class_map[self.run_config.model][VariableNames.AlphaStar]
        model_obj_alpha = model_class_alpha(gamma=gamma, T=self.run_config.T)
        alpha_star = model_obj_alpha.get_alpha_star(Xs=Xs, Ss=Ss, Is=Is)
        # if conf.is_simulation:
        #     alpha_star[alpha_star < 0] = 0
        #     alpha_star[alpha_star > 1] = 1
        # alpha_star = np.array(len(alpha_star)*[1.25])
        return alpha_star

    def get_I_star(self, alpha_star, Xs, Ss):
        if self.run_config.seed is not None:
            np.random.seed(self.run_config.seed)
        model_class_I = model_class_map[self.run_config.model][VariableNames.IStar]
        model_obj_I = model_class_I(alpha_star=alpha_star)
        I_star = model_obj_I.get_I_star(Xs=Xs, Ss=Ss)
        return I_star

    def get_I_star_utility_dict(self, gamma, Is, Xs, Ss):
        I = pd.DataFrame()
        Utility = pd.DataFrame()
        if self.run_config.is_include_optimal_control:
            alpha_star = self.get_alpha_star(gamma=gamma, Xs=Xs, Ss=Ss, Is=Is)
            # todo: add back alpha rate range
            # if type(alpha_star) is np.ndarray:
            #     alpha_star[alpha_star < 0] = model_params.eps
            #     alpha_star[alpha_star > 1] = 1
            I_optimal = self.get_I_star(alpha_star=alpha_star, Xs=Xs, Ss=Ss)
            I[f'Optimal Control'] = I_optimal
            Utility[f'Optimal Control'] = Driver.utility(I=I_optimal, gamma=gamma)
        if self.run_config.is_include_full_control:
            I_full = self.get_I_star(alpha_star=1, Xs=Xs, Ss=Ss)
            I[f'Full Control'] = I_full
            Utility[f'Full Control'] = Driver.utility(I=I_full, gamma=gamma)
        if self.run_config.is_include_no_control:
            I_no = self.get_I_star(alpha_star=model_params.eps, Xs=Xs, Ss=Ss)
            I['No Control'] = I_no
            Utility[f'No Control'] = Driver.utility(I=I_no, gamma=gamma)
        return I, Utility

    def get_expected_I_star_utility_dict(self, gamma, data_elm):
        Xs, Ss, Is = data_elm
        # dIs = Is[1:] - Is[:-1]
        # dSs = Ss[1:] - Ss[:-1]
        # dXs = Xs[1:] - Xs[:-1]
        with Pool(self.run_config.cpu) as p:
            ret = list(
                tqdm(p.imap(partial(self.get_I_star_utility_dict, Xs=Xs, Ss=Ss, Is=Is),
                            [gamma] * self.run_config.n_trials_monte_carlo), total=self.run_config.n_trials_monte_carlo))
        I_list = [item[0] for item in ret]
        Utility_list = [item[1] for item in ret]
        I = reduce(lambda a, b: a.add(b, fill_value=0), I_list)
        I = I / self.run_config.n_trials_monte_carlo
        Utility = reduce(lambda a, b: a.add(b, fill_value=0), Utility_list)
        Utility = Utility / self.run_config.n_trials_monte_carlo
        return I, Utility

    # def get_I_star_utility_dict_simulation(self, gamma, data):
    #     """
    #     Get all lines for a given gamma
    #     data: [(data.Xs_trials, data.Ss_trials , data.Is_trials), ... ]. Length is 1 for monte carlo
    #     """
    #     paralell = False
    #     I_list = []
    #     Utility_list = []
    #     if paralell:
    #         with Pool(self.run_config.cpu) as p:
    #             ret = list(tqdm(p.imap(partial(self.simulation_parallel_helper, gamma=gamma), data),
    #                             total=self.run_config.n_trials_data_generation))
    #         I_list = [item[0] for item in ret]
    #         Utility_list = [item[1] for item in ret]
    #     else:
    #         for data_elm in data:
    #             I, Utility = self.simulation_parallel_helper(data_elm=data_elm, gamma=gamma)
    #             I_list.append(I)
    #             Utility_list.append(Utility)
    #
    #     I = reduce(lambda a, b: a.add(b, fill_value=0), I_list)
    #     I = I / self.run_config.n_trials_data_generation
    #     Utility = reduce(lambda a, b: a.add(b, fill_value=0), Utility_list)
    #     Utility = Utility / self.run_config.n_trials_data_generation
    #     return I, Utility

    def run(self):
        # generate simulated data
        data_simulation = self.simulation_dict[self.run_config.model]
        data = data_simulation(I0=self.run_config.I0, X0=self.run_config.X0, S0=self.run_config.S0,
                               n_steps=self.run_config.n_steps_simulation_data_generation,
                               n_trials=self.run_config.n_trials_simulation_data_generation)
        if not hasattr(data, 'Ss_trials'):
            data.Ss_trials = np.zeros((self.run_config.n_trials_simulation_data_generation, self.run_config.n_steps_simulation_data_generation))
        if not hasattr(data, 'Is_trials'):
            data.Is_trials = np.zeros((self.run_config.n_trials_simulation_data_generation, self.run_config.n_steps_simulation_data_generation))
        dataset = list(zip(list(data.Xs_trials), list(data.Ss_trials), list(data.Is_trials)))

        # simulate for no control
        gamma_to_results = {}
        for gamma in self.run_config.gammas:
            if self.run_config.is_simulation:
                I, Utility = self.get_expected_I_star_utility_dict(gamma=gamma, data_elm=dataset[0])
            else:
                Xs = np.array(conf.real_world_data['X(t)'])
                # ts = np.array(conf.real_world_data['t'])
                Is = np.array(conf.real_world_data['I(t)'])
                Ss = np.array(conf.real_world_data['S(t)'])
                I, Utility = self.get_I_star_utility_dict(
                    gamma=gamma,
                    Is=Is,
                    Xs=Xs,
                    Ss=Ss
                )
            gamma_to_results[gamma] = (I, Utility)
        # simulate for full control

        # simulate for optimal control

        # plot
        plot_generator = PlotGenerator()
        plot_generator.plot(
            gammas=self.run_config.gammas,
            gamma_to_results=gamma_to_results,
            is_simulation=self.run_config.is_simulation,
            model=self.run_config.model
        )
