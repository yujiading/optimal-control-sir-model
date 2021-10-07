from library.run_config import RunConfig
import pickle
from library.plot_generator import PlotGenerator
from library.conf import cur_dir_path
from library.models.model_mapper import ModelTypes
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes

from library.models.model_result import ModelResult

def test_plot_real_data():
    fig, axes = plt.subplots(nrows=6, ncols=4)
    super_title = f"Infection $I(t)$ of One Scenario Real Data"
    model_classes = ['ModelTypes.LowConst', 'ModelTypes.LowOU', 'ModelTypes.ModerateConst', 'ModelTypes.ModerateOU']
    model_names = ['Low Infection\nRegime with\nConstant Treatment',
                   'Low Infection\nRegime with\nOU Treatment',
                   'Moderate Infection\nRegime with\nConstant Treatment',
                   'Moderate Infection\nRegime with\nOU Treatment'
                   ]
    for j, model in enumerate(model_classes):
        run_config = RunConfig(model=eval(model))

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

        axes[0,j].set_title(model_names[j], pad=15,fontsize= 10)
        gammas = run_config.gammas
        for i, gamma in enumerate(gammas):
            results_dict = average_gamma_to_results[gamma]
            result_keys = ['Optimal Control', 'Full Control', 'No Control']
            result_key_to_line_style = {
                'Optimal Control': '-',
                'Full Control': ':',
                'No Control': '-.'
            }
            for result_key in result_keys:
                model_result: ModelResult = results_dict[result_key]
                axes[i, j].plot(
                    model_result.average_simulation_result.Is,
                    label=result_key,
                    linestyle=result_key_to_line_style[result_key]
                )
                # axes[i, 0].set_yscale('symlog')

            axes[i, 0].set_ylabel('$\\gamma=$' + str(gamma))
            yfmt = ScalarFormatterForceFormat()
            yfmt.set_powerlimits((0, 0))
            axes[i, j].yaxis.set_major_formatter(yfmt)
            yfmt = ScalarFormatterForceFormat()

        handles, labels = axes[0, 0].get_legend_handles_labels()

        # Format plot
        fig.set_size_inches(8, 10.5)

        fig.subplots_adjust(left=0.1, bottom=0.13, right=0.95, top=0.88, wspace=0.3, hspace=0.6)

        fig.legend(handles, labels, bbox_to_anchor=(0.5, 0.025), loc='lower center')
        plt.suptitle(f'{super_title}', x=0.5)
    plt.show()

class ScalarFormatterForceFormat(ScalarFormatter):
    pass
    # def _set_format(self):  # Override function that finds format to use.
    #     self.format = "%1.1f"  # Give format here
