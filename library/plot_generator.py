from functools import partial, reduce
from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import ScalarFormatter
from tqdm import tqdm
from library.models.model_result import ModelResult
from library import conf
from library.I_star import IStarLowConst, IStarLowOU, IStarModerateOU, IStarModerateConst
from library.alpha_star import AlphaStarLowConst, AlphaStarLowOU, AlphaStarModerateOU, AlphaStarModerateConst
from library.data_simulation import DataModerateOU, DataLowOU, DataLowConst, DataModerateConst
from models.model_mapper import ModelTypes, VariableNames, model_class_map
from typing import List, Dict


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

        for i, gamma in enumerate(gammas):
            I, Utility = gamma_to_results[gamma]

            I.plot(ax=axes[i, 0], style=styles, legend=False, title=subtitle_I, sharex=True)
            axes[i, 0].set_ylabel('$\\gamma=$' + str(gamma))
            Utility.plot(ax=axes[i, 1], style=styles, legend=False, title=subtitle_utility)
            yfmt = ScalarFormatterForceFormat()
            yfmt.set_powerlimits((0, 0))
            axes[i, 0].yaxis.set_major_formatter(yfmt)
            yfmt = ScalarFormatterForceFormat()
            yfmt.set_powerlimits((0, 0))
            axes[i, 1].yaxis.set_major_formatter(yfmt)

            subtitle_I = None
            subtitle_utility = None

        handles, labels = axes[0, 0].get_legend_handles_labels()

        # Format plot
        fig.set_size_inches(8, 10.5)
        fig.subplots_adjust(left=0.1, bottom=0.2, right=0.95, top=0.9, wspace=0.3, hspace=0.4)
        fig.legend(handles, labels, bbox_to_anchor=(0.5, 0.025), loc='lower center')
        plt.suptitle(f'{infection_type} Infection Regime with {treatment_type} Treatment', x=0.5)
        plt.show()


class ScalarFormatterForceFormat(ScalarFormatter):
    pass
    # def _set_format(self):  # Override function that finds format to use.
    #     self.format = "%1.1f"  # Give format here
