from typing import Type

import numpy as np
import pandas as pd
from tqdm import tqdm

from library import conf
from library.models.base_simulator import BaseSimulator
from library.models.low_simulators import LowOUSimulator, LowConstSimulator
from library.models.model_result import ModelResult
from library.models.moderate_simulators import ModerateOUSimulator, ModerateConstSimulator
from models import model_params
from models.model_mapper import ModelTypes
from plot_generator import PlotGenerator
from run_config import RunConfig


class Driver:
    def __init__(self, run_config: RunConfig):
        """
        requires X0, I0, S0, cpu, n_trials, n_steps if is_simulation is True
        """
        self.run_config = run_config

        self.simulation_dict = {
            "LowConst": LowConstSimulator,
            "LowOU": LowOUSimulator,
            "ModerateConst": ModerateConstSimulator,
            "ModerateOU": ModerateOUSimulator
        }

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

    def run(self):
        all_gamma_to_results = []
        for i in tqdm(range(self.run_config.n_trials_simulated_data_generation)):
            # get dataset for modeling Xs, Is, Ss
            if self.run_config.is_simulation:
                simulator_class: Type[BaseSimulator] = self.simulation_dict[self.run_config.model]
                simulator = simulator_class(gamma=-1, run_config=self.run_config)
                simulation_result = simulator.simulate_one_trial(alpha_star=np.array(
                    [self.run_config.alpha_fix] * self.run_config.n_steps_simulated_data_generation))
                Xs = simulation_result.Xs
                Is = simulation_result.Is
                Ss = simulation_result.Ss
            else:
                Xs = np.array(conf.real_world_data['X(t)'])
                # ts = np.array(conf.real_world_data['t'])
                Is = np.array(conf.real_world_data['I(t)'])
                Ss = np.array(conf.real_world_data['S(t)'])

            gamma_to_results = {}
            for gamma in tqdm(self.run_config.gammas):
                # simulate for no control
                simulator_class: Type[BaseSimulator] = self.simulation_dict[self.run_config.model]
                simulator = simulator_class(gamma=gamma, run_config=self.run_config)

                # simulate for no control
                alpha_star = np.array([0.0001] * len(Xs))
                no_control_simulation_result, _ = simulator.run_monte_carlo_simulation(alpha_star=alpha_star)
                no_control_model_result = ModelResult(
                    model_type=self.run_config.model,
                    alpha_star=alpha_star,
                    average_simulation_result=no_control_simulation_result
                )

                # simulate for full control
                alpha_star = np.array([1] * len(Xs))
                full_control_simulation_result, _ = simulator.run_monte_carlo_simulation(alpha_star=alpha_star)
                full_control_model_result = ModelResult(
                    model_type=self.run_config.model,
                    alpha_star=alpha_star,
                    average_simulation_result=full_control_simulation_result
                )

                # simulate for optimal control
                optimal_control_model_result = simulator.run_model_and_monte_carlo_simulation(
                    Xs=Xs, Ss=Ss, Is=Is)

                # combine results for gamma
                gamma_to_results[gamma] = {
                    'Optimal Control': optimal_control_model_result,
                    'Full Control': full_control_model_result,
                    'No Control': no_control_model_result
                }
            all_gamma_to_results.append(gamma_to_results)

        result_keys = ['Optimal Control', 'Full Control', 'No Control']
        average_gamma_to_results = {}
        for gamma in self.run_config.gammas:
            gamma_to_result = {}
            for result_key in result_keys:
                all_model_results = [all_gamma_to_result[gamma][result_key] for all_gamma_to_result in
                                     all_gamma_to_results]
                gamma_to_result[result_key] = ModelResult.average_model_results(all_model_results)
            average_gamma_to_results[gamma] = gamma_to_result

        # plot
        plot_generator = PlotGenerator()
        plot_generator.plot(
            gammas=self.run_config.gammas,
            gamma_to_results=average_gamma_to_results,
            is_simulation=self.run_config.is_simulation,
            model=self.run_config.model
        )


def run():
    """
    model takes values: 'LowConst', 'LowOU', 'ModerateConst', 'ModerateOU'
    """
    run_config = RunConfig(model=ModelTypes.LowConst)
    driver = Driver(run_config=run_config)
    driver.run()

    run_config = RunConfig(model=ModelTypes.LowOU)
    driver = Driver(run_config=run_config)
    driver.run()

    run_config = RunConfig(model=ModelTypes.ModerateConst)
    driver = Driver(run_config=run_config)
    driver.run()

    run_config = RunConfig(model=ModelTypes.ModerateOU)
    driver = Driver(run_config=run_config)
    driver.run()


if __name__ == '__main__':
    run()
