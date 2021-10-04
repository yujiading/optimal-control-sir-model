# from library.I_star import IStarLowConst, IStarLowOU, IStarModerateOU, IStarModerateConst
# from library.alpha_star import AlphaStarLowConst, AlphaStarLowOU, AlphaStarModerateOU, AlphaStarModerateConst
# from library.data_simulation import DataModerateOU, DataLowOU, DataLowConst, DataModerateConst


class ModelTypes:
    LowConst = 'LowConst'
    LowOU = 'LowOU'
    ModerateConst = 'ModerateConst'
    ModerateOU = 'ModerateOU'


class VariableNames:
    AlphaStar = 'AlphaStar'
    IStar = 'IStar'


# model_class_map = {
#     ModelTypes.LowConst: {
#         VariableNames.AlphaStar: AlphaStarLowConst,
#         VariableNames.IStar: IStarLowConst
#     },
#     ModelTypes.LowOU: {
#         VariableNames.AlphaStar: AlphaStarLowOU,
#         VariableNames.IStar: IStarLowOU
#     },
#     ModelTypes.ModerateConst: {
#         VariableNames.AlphaStar: AlphaStarModerateConst,
#         VariableNames.IStar: IStarModerateConst
#     },
#     ModelTypes.ModerateOU: {
#         VariableNames.AlphaStar: AlphaStarModerateOU,
#         VariableNames.IStar: IStarModerateOU
#     }
# }
