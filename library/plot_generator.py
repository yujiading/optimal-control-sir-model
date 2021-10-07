import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

from library.models.model_result import ModelResult
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

class PlotGenerator:
    treatment_type_dict = {"LowConst": "Constant",
                           "LowOU": "OU",
                           "ModerateConst": "Constant",
                           "ModerateOU": "OU"}
    infection_type_dict = {"LowConst": "Low",
                           "LowOU": "Low",
                           "ModerateConst": "Moderate",
                           "ModerateOU": "Moderate"}

    def plot(
            self,
            gammas,
            gamma_to_results,
            is_simulation,
            model
    ):
        infection_type = self.infection_type_dict[model]
        treatment_type = self.treatment_type_dict[model]
        # styles = ['C0o-.', 'C1*:', 'C2<-.', 'C3>-.', 'C4^-.', 'C5-', 'C6--']
        styles = ['C0-', 'C1:', 'C2-.', 'C3-.', 'C4-.', 'C5-', 'C6--']
        fig, axes = plt.subplots(nrows=len(gammas), ncols=2)
        if is_simulation:
            subtitle_I = f"Expected Infection: $EI(t)$"
            subtitle_utility = "Expected Utility: $E-\\frac{I(t)^{1-\\gamma}}{1-\\gamma}$"
        else:
            subtitle_I = f"Infection: $I(t)$"
            subtitle_utility = "Utility: $-\\frac{I(t)^{1-\\gamma}}{1-\\gamma}$"
        axes[0, 0].set_title(subtitle_I, pad=15)
        axes[0, 1].set_title(subtitle_utility, pad=15)
        for i, gamma in enumerate(gammas):
            results_dict = gamma_to_results[gamma]
            result_keys = ['Optimal Control', 'Full Control', 'No Control']
            result_key_to_line_style = {
                'Optimal Control': '-',
                'Full Control': ':',
                'No Control': '-.'
            }
            for result_key in result_keys:
                model_result: ModelResult = results_dict[result_key]
                axes[i, 0].plot(
                    model_result.average_simulation_result.Is,
                    label=result_key,
                    linestyle=result_key_to_line_style[result_key]
                )
                # axes[i, 0].set_yscale('symlog')
                axes[i, 1].plot(
                    model_result.average_simulation_result.Utility,
                    label=result_key,
                    linestyle=result_key_to_line_style[result_key]
                )
                # axes[i, 1].set_yscale('symlog')

            axes[i, 0].set_ylabel('$\\gamma=$' + str(gamma))
            yfmt = ScalarFormatterForceFormat()
            yfmt.set_powerlimits((0, 0))
            axes[i, 0].yaxis.set_major_formatter(yfmt)
            yfmt = ScalarFormatterForceFormat()
            yfmt.set_powerlimits((0, 0))
            axes[i, 1].yaxis.set_major_formatter(yfmt)

            is_zoom_out_plots=True
            if is_zoom_out_plots:
                axins = zoomed_inset_axes(axes[i, 1], 2, loc='right',borderpad=-9)  # zoom = 2
                for result_key in result_keys:
                    model_result: ModelResult = results_dict[result_key]
                    axins.plot(model_result.average_simulation_result.Utility,
                        label=result_key,
                        linestyle=result_key_to_line_style[result_key])
                bottom, top = axins.get_ylim()
                axins.set_xlim(80, 100)
                axins.set_ylim(bottom, top-(top-bottom)/2)
                plt.xticks(visible=False)
                plt.yticks(visible=False)
                mark_inset(axes[i, 1], axins, loc1=2, loc2=4, fc="none", ec="0.5")
                plt.draw()

            subtitle_I = None
            subtitle_utility = None

        handles, labels = axes[0, 0].get_legend_handles_labels()

        # Format plot
        fig.set_size_inches(8, 10.5)
        if is_zoom_out_plots:
            fig.subplots_adjust(left=0.1, bottom=0.13, right=0.8, top=0.92, wspace=0.3, hspace=0.6)
        else:
            fig.subplots_adjust(left=0.1, bottom=0.13, right=0.95, top=0.92, wspace=0.3, hspace=0.6)

        fig.legend(handles, labels, bbox_to_anchor=(0.5, 0.025), loc='lower center')
        plt.suptitle(f'{infection_type} Infection Regime with {treatment_type} Treatment', x=0.5)
        plt.show()


class ScalarFormatterForceFormat(ScalarFormatter):
    pass
    # def _set_format(self):  # Override function that finds format to use.
    #     self.format = "%1.1f"  # Give format here
