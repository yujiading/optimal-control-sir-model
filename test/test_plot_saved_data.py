from library.run_config import RunConfig
import pickle
from library.plot_generator import PlotGenerator
from library.conf import cur_dir_path


def test_plot_saved_data():
    run_config = RunConfig()
    params = (
        run_config.model,
        run_config.gammas,
        run_config.n_steps_simulated_data_generation,
        run_config.n_trials_simulated_data_generation,
        run_config.n_trials_monte_carlo_simulation,
    )
    params_str = '_'.join([str(param) for param in params])
    filehandler = open(cur_dir_path / f"../data/average_gamma_to_results_{params_str}.pickle", "rb")
    average_gamma_to_results = pickle.load(filehandler)
    filehandler.close()

    plot_generator = PlotGenerator()
    plot_generator.plot(
        gammas=run_config.gammas,
        gamma_to_results=average_gamma_to_results,
        is_simulation=run_config.is_simulation,
        model=run_config.model
    )
