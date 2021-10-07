import math

from library.sigma_s import SigmaSDoubleSum, SigmaSSquareConst


def test_sigmas_double_sum_one_scenario():
    sigma_s = SigmaSDoubleSum()
    sigma_s_lst, sigma_s_square_lst = sigma_s.simulate_one_scenario
    print(sum(sigma_s_lst) / len(sigma_s_lst), math.sqrt(sum(sigma_s_square_lst) / len(sigma_s_square_lst)))

def test_sigmas_double_sum_all_scenarios():
    sigma_s = SigmaSDoubleSum(scenarios=10000)
    sigma_s_all_lst, sigma_s_square_all_lst = sigma_s.simulate_all_scenarios
    print(sum(sigma_s_all_lst)/len(sigma_s_all_lst), math.sqrt(sum(sigma_s_square_all_lst) / len(sigma_s_square_all_lst)))

def test_sigmas_square_const():
    sigmas = SigmaSSquareConst().sigma_s
    print(sigmas)

def test_sigmas_all():
    sigmas = SigmaSSquareConst().sigma_s
    print(f"top: {sigmas}")
    sigma_s = SigmaSDoubleSum(scenarios=10000)
    sigma_s_all_lst, sigma_s_square_all_lst = sigma_s.simulate_all_scenarios
    print(f"middle {sum(sigma_s_all_lst) / len(sigma_s_all_lst)}")
    print(f"bottom {math.sqrt(sum(sigma_s_square_all_lst) / len(sigma_s_square_all_lst))}")
