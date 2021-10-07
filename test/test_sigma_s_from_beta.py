import matplotlib.pyplot as plt

from library.sigma_s_from_beta import SigmaSFromBeta


def test_monte_carlo_one_scenario():
    sigma_s = SigmaSFromBeta()
    S_series = sigma_s.monte_carlo_one_scenario(beta_change=0)
    plt.plot(S_series)
    plt.show()


def test_monte_carlo_all_scenarios():
    sigma_s = SigmaSFromBeta()
    S_all_series = sigma_s.monte_carlo_all_scenarios
    sigma_s.plot_all_scenarios(S_all_series=S_all_series)


def test_get_std_of_normal_given_prob():
    std = SigmaSFromBeta.get_std_of_normal_given_prob(low=67.36, up=72.64, mean=70, percent=0.34)
    print(std)


def test_sigma_s_one_estimate():
    sigma_s = SigmaSFromBeta(trial_length=10)
    S_all_series = sigma_s.monte_carlo_all_scenarios
    sigma_s_lst = []
    for i in range(1, sigma_s.trial_length + 1):
        sigma_s_ = sigma_s.sigma_s_one_estimate(S_all_series=S_all_series, trail_index=i, percent=0.67)
        sigma_s_lst.append(sigma_s_)
        # print(sigma_s_)
    print(sum(sigma_s_lst) / len(sigma_s_lst))
