from library.run_config import RunConfig
import pickle
from library.plot_generator import PlotGenerator
from library.conf import cur_dir_path
from library.models.model_mapper import ModelTypes

def test_plot_real_data():
    run_config = RunConfig(model=ModelTypes.LowConst,
                           is_simulation=False,
                           n_trials_simulated_data_generation=1,
                           n_trials_monte_carlo_simulation=1
                           )


    plot_generator = PlotGenerator()
    plot_generator.plot(
        gammas=run_config.gammas,
        gamma_to_results=average_gamma_to_results,
        is_simulation=run_config.is_simulation,
        model=run_config.model
    )