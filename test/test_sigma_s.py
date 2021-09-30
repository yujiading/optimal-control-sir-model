from library.sigma_s import SigmaSDoubleSum


def test_sigmas_double_sum_one_scenario():
    sigma_s = SigmaSDoubleSum()
    sigma_s_lst = sigma_s.simulate_one_scenario
    print(sum(sigma_s_lst) / len(sigma_s_lst))

def test_simulate_all_scenarios():
    sigma_s = SigmaSDoubleSum(scenarios=10000)
    sigma_s_all_lst = sigma_s.simulate_all_scenarios
    print(sum(sigma_s_all_lst)/len(sigma_s_all_lst))
