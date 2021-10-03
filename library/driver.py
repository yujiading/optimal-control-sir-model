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

    def get_alpha_star(self, gamma):
        model_class = model_class_map[self.run_config.model][VariableNames.AlphaStar]
        cla_alpha = model_class(gamma=gamma, T=self.run_config.T)
        alpha_star = cla_alpha.alpha_star
        # if conf.is_simulation:
        #     alpha_star[alpha_star < 0] = 0
        #     alpha_star[alpha_star > 1] = 1
        # alpha_star = np.array(len(alpha_star)*[1.25])
        return alpha_star

    def get_I_star(self, alpha_star):
        model_class = model_class_map[self.run_config.model][VariableNames.IStar]
        cla_I = model_class(alpha_star=alpha_star)
        I_star = cla_I.I_star
        return I_star

    def get_I_star_utility_dict(self, gamma):
        I = pd.DataFrame()
        Utility = pd.DataFrame()
        if self.run_config.is_include_optimal_control:
            alpha_star = self.get_alpha_star(gamma=gamma)
            I_optimal = self.get_I_star(alpha_star=alpha_star)
            I[f'Optimal Control'] = I_optimal
            Utility[f'Optimal Control'] = Driver.utility(I=I_optimal, gamma=gamma)
        if self.run_config.is_include_full_control:
            I_full = self.get_I_star(alpha_star=1)
            I[f'Full Control'] = I_full
            Utility[f'Full Control'] = Driver.utility(I=I_full, gamma=gamma)
        if self.run_config.is_include_no_control:
            I_no = self.get_I_star(alpha_star=0)
            I['No Control'] = I_no
            Utility[f'No Control'] = Driver.utility(I=I_no, gamma=gamma)
        return I, Utility

    def get_expected_I_star_utility_dict(self, gamma):
        with Pool(self.run_config.cpu) as p:
            ret = list(
                tqdm(p.imap(partial(self.get_I_star_utility_dict), [gamma] * self.run_config.n_trials), total=self.run_config.n_trials))
        I_list = [item[0] for item in ret]
        Utility_list = [item[1] for item in ret]
        I = reduce(lambda a, b: a.add(b, fill_value=0), I_list)
        I = I / self.run_config.n_trials
        Utility = reduce(lambda a, b: a.add(b, fill_value=0), Utility_list)
        Utility = Utility / self.run_config.n_trials
        return I, Utility

    def simulation_parallel_helper(self, data_elm, gamma):
        """
        data: [(Xs_trials, Ss_trials, Is_trials), ...]
        """
        conf.Xs, conf.Ss, conf.Is = data_elm
        conf.length = len(conf.Is)
        conf.dIs = conf.Is[1:] - conf.Is[:-1]
        conf.dSs = conf.Ss[1:] - conf.Ss[:-1]
        conf.dXs = conf.Xs[1:] - conf.Xs[:-1]
        I, Utility = self.get_expected_I_star_utility_dict(gamma=gamma)
        return I, Utility

    def get_I_star_utility_dict_simulation(self, gamma, data):
        """
        Get all lines for a given gamma
        data: [(data.Xs_trials, data.Ss_trials , data.Is_trials), ... ]. Length is 1 for monte carlo
        """
        paralell = False
        I_list = []
        Utility_list = []
        if paralell:
            with Pool(self.run_config.cpu) as p:
                ret = list(tqdm(p.imap(partial(self.simulation_parallel_helper, gamma=gamma), data),
                                total=self.run_config.n_trials_data_generation))
            I_list = [item[0] for item in ret]
            Utility_list = [item[1] for item in ret]
        else:
            for data_elm in data:
                I, Utility = self.simulation_parallel_helper(data_elm=data_elm, gamma=gamma)
                I_list.append(I)
                Utility_list.append(Utility)

        I = reduce(lambda a, b: a.add(b, fill_value=0), I_list)
        I = I / self.run_config.n_trials_data_generation
        Utility = reduce(lambda a, b: a.add(b, fill_value=0), Utility_list)
        Utility = Utility / self.run_config.n_trials_data_generation
        return I, Utility

    def run(self):
        # generate simulated data
        data_simulation = self.simulation_dict[self.run_config.model]
        data = data_simulation(I0=self.run_config.I0, X0=self.run_config.X0, S0=self.run_config.S0, n_steps=self.run_config.n_steps,
                               n_trials=self.run_config.n_trials_data_generation)
        if not hasattr(data, 'Ss_trials'):
            data.Ss_trials = np.zeros((self.run_config.n_trials_data_generation, self.run_config.n_steps))
        if not hasattr(data, 'Is_trials'):
            data.Is_trials = np.zeros((self.run_config.n_trials_data_generation, self.run_config.n_steps))
        dataset = list(zip(list(data.Xs_trials), list(data.Ss_trials), list(data.Is_trials)))

        # simulate for no control
        gamma_to_results = {}
        for gamma in self.run_config.gammas:
            if self.run_config.is_simulation:
                I, Utility = self.get_I_star_utility_dict_simulation(gamma=gamma, data=dataset)
            else:
                I, Utility = self.get_I_star_utility_dict(gamma=gamma)
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
