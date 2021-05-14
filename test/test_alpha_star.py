from library.alpha_star import AlphaStarLowConst


def test_alpha_star_low_const():
    cla = AlphaStarLowConst(gamma=-5)
    print(cla.alpha_star)
