import torch
from LOLA_dice import Agent # needed
from matplotlib import pyplot as plt
import numpy as np

def load_from_checkpoint(ckpt_path, newckpt = False):
    assert ckpt_path is not None
    print(f"loading model from {ckpt_path}")
    ckpt_dict = torch.load(ckpt_path, map_location=torch.device('cpu'))
    agent1 = ckpt_dict["agent1"]
    agent2 = ckpt_dict["agent2"]
    info = ckpt_dict["info"]
    if not newckpt:
        return agent1, agent2, info
    else:
        scores = ckpt_dict["scores"]

        return agent1, agent2, info, scores


def load_from_checkpoint_new(ckpt_path):
    assert ckpt_path is not None
    print(f"loading model from {ckpt_path}")
    ckpt_dict = torch.load(ckpt_path, map_location=torch.device('cpu'))
    agent1 = ckpt_dict["agent1"]
    agent2 = ckpt_dict["agent2"]
    info = ckpt_dict["info"]
    scores = ckpt_dict["scores"]
    vs_fixed_strat_scores = ckpt_dict["vs_fixed_scores"]
    return agent1, agent2, info, scores, vs_fixed_strat_scores

file_path = "."

plot_coin = False # True

if plot_coin:

    ckpts_pola = ["checkpoint_200_2022-05-08_18-39_seed10.pt",
                  "checkpoint_200_2022-05-08_15-15_seed11.pt",
                  "checkpoint_200_2022-05-08_15-24_seed12.pt",
                  "checkpoint_200_2022-05-09_03-41_seed13.pt",
                  "checkpoint_200_2022-05-09_03-32_seed14.pt",
                  "checkpoint_200_2022-05-11_06-43_seed7.pt",
                  "checkpoint_200_2022-05-11_00-39_seed8.pt",
                  "checkpoint_200_2022-05-11_07-56_seed9.pt",
                  "checkpoint_200_2022-05-12_02-42_seed5.pt",
                  "checkpoint_200_2022-05-12_02-42_seed6.pt"]

    ckpts_lola = [
                  ["checkpoint_5000_2022-05-08_23-27_seed10.pt", "checkpoint_2000_2022-05-09_17-07_seed10.pt", "checkpoint_5000_2022-05-10_20-15_seed10.pt"],
                  ["checkpoint_5000_2022-05-08_23-26_seed11.pt", "checkpoint_2000_2022-05-09_17-03_seed11.pt", "checkpoint_8000_2022-05-10_09-55_seed11.pt"],
                  ["checkpoint_5000_2022-05-08_22-56_seed12.pt", "checkpoint_2000_2022-05-09_17-03_seed12.pt", "checkpoint_5000_2022-05-10_20-16_seed12.pt"],
                  "checkpoint_12000_2022-05-10_08-50_seed3.pt",
                  "checkpoint_12000_2022-05-10_14-25_seed1.pt",
                  "checkpoint_12000_2022-05-10_14-42_seed2.pt",
                  "checkpoint_14000_2022-05-11_05-03_seed4.pt",
                  "checkpoint_14000_2022-05-11_05-06_seed5.pt",
                  "checkpoint_14000_2022-05-11_05-20_seed6.pt",
                  "checkpoint_14000_2022-05-11_05-29_seed7.pt" # There is kind of an arbitrary choice of seeds based on runs that got interrupted and based on which seeds I remembered having done vs not having done
                  ]


    ckpts_pola_om = ["checkpoint_175_2022-05-10_14-38_seed6.pt",
                     "checkpoint_175_2022-05-10_10-56_seed7.pt",
                     "checkpoint_175_2022-05-10_09-51_seed8.pt",
                     "checkpoint_200_2022-05-11_11-08_seed1.pt",
                     "checkpoint_200_2022-05-11_10-48_seed2.pt",
                     "checkpoint_200_2022-05-11_11-06_seed3.pt",
                     "checkpoint_200_2022-05-11_11-00_seed4.pt",
                     "checkpoint_175_2022-05-11_12-28_seed5.pt",
                     "checkpoint_200_2022-05-12_12-51_seed9.pt",
                     "checkpoint_200_2022-05-12_13-04_seed10.pt"]

else:
    # PLOT IPD
    ckpts_pola = ["checkpoint_250_2022-05-05_12-13.pt", # seed 3
                  "checkpoint_250_2022-05-05_12-16.pt", # seed 4
                  "checkpoint_250_2022-05-05_12-19.pt", # seed 5
                  ["checkpoint_100_2022-05-06_17-54.pt", "checkpoint_50_2022-05-07_03-47.pt"], # seed 1
                  ["checkpoint_100_2022-05-06_17-51.pt", "checkpoint_50_2022-05-07_03-48.pt"], # seed 2
                  ["checkpoint_100_2022-05-06_17-49.pt", "checkpoint_50_2022-05-07_03-48_seed6.pt"], # seed 6
                  ["checkpoint_100_2022-05-06_17-57.pt", "checkpoint_50_2022-05-07_03-40.pt"],  # seed 7
                  "checkpoint_200_2022-05-12_23-55_seed13.pt", # runs that didn't get interrupted
                  "checkpoint_200_2022-05-13_00-22_seed14.pt",
                  "checkpoint_200_2022-05-13_00-28_seed15.pt",
                  ]
    ckpts_lola = ["checkpoint_10000_2022-05-04_00-00.pt",
                  "checkpoint_10000_2022-05-04_00-03.pt",
                  "checkpoint_10000_2022-05-03_23-42.pt",
                  "checkpoint_15000_2022-05-12_09-44_seed10.pt",
                  "checkpoint_15000_2022-05-12_04-21_seed11.pt",
                  "checkpoint_15000_2022-05-12_04-22_seed12.pt",
                  "checkpoint_15000_2022-05-12_04-33_seed13.pt",
                  "checkpoint_15000_2022-05-15_00-46_seed14.pt",
                  "checkpoint_15000_2022-05-14_18-24_seed15.pt",
                  "checkpoint_15000_2022-05-14_16-20_seed16.pt"
                  ]
    ckpts_pola_om = ["checkpoint_150_2022-05-05_10-03.pt", # seed 3
                     "checkpoint_150_2022-05-05_10-05.pt", # seed 4
                     "checkpoint_150_2022-05-05_10-07.pt", # seed 5
                     ["checkpoint_50_2022-05-06_16-33.pt", "checkpoint_50_2022-05-07_04-35.pt"],  # seed 1
                     ["checkpoint_50_2022-05-06_16-34.pt", "checkpoint_50_2022-05-06_23-40.pt", "checkpoint_50_2022-05-07_04-32.pt"],  # seed 2
                     ["checkpoint_50_2022-05-06_16-35_seed6.pt", "checkpoint_50_2022-05-07_04-27.pt"],  # seed 6
                     ["checkpoint_50_2022-05-06_16-35_seed7.pt", "checkpoint_50_2022-05-07_04-23.pt"],  # seed 7
                     ["checkpoint_200_2022-05-13_03-48_seed8.pt"],
                     "checkpoint_200_2022-05-13_07-19_seed9.pt",
                     "checkpoint_200_2022-05-13_09-15_seed10.pt",
                     ]
    ckpts_static = [
                    "checkpoint_2_2022-05-15_17-47_seed1.pt",
                    "checkpoint_2_2022-05-15_17-47_seed2.pt",
                    "checkpoint_2_2022-05-15_17-48_seed3.pt",
                    "checkpoint_2_2022-05-15_17-48_seed4.pt",
                    "checkpoint_2_2022-05-15_17-48_seed5.pt",
                    "checkpoint_2_2022-05-15_17-44_seed6.pt",
                    "checkpoint_2_2022-05-15_17-45_seed7.pt",
                    "checkpoint_2_2022-05-15_17-45_seed8.pt",
                    "checkpoint_2_2022-05-15_17-46_seed9.pt",
                    "checkpoint_2_2022-05-15_17-46_seed10.pt",
                    ]


def get_prop_same_coins(ckpts, max_iter_plot=200):
    prop_same_coins_record = []
    for i in range(len(ckpts)):
        full_path = f"{file_path}/{ckpts[i]}"
        agent1, agent2, info = load_from_checkpoint(full_path)
        same_colour_coins_record, diff_colour_coins_record = info
        same_colour_coins_record = torch.FloatTensor(same_colour_coins_record[:max_iter_plot])
        diff_colour_coins_record = torch.FloatTensor(diff_colour_coins_record[:max_iter_plot])
        prop_same_coins = same_colour_coins_record / (same_colour_coins_record + diff_colour_coins_record)
        prop_same_coins_record.append(prop_same_coins.clone())
    return torch.stack(prop_same_coins_record)


def get_score_individual_ckpt(full_path, w_coin_record=False):
    agent1, agent2, info, scores, vs_fixed_strat_scores = load_from_checkpoint_new(
        full_path)

    agent1_vs_fixed_strat_scores, agent2_vs_fixed_strat_scores = vs_fixed_strat_scores
    agent1_vs_fixed_strat_scores = torch.stack(agent1_vs_fixed_strat_scores)
    agent2_vs_fixed_strat_scores = torch.stack(agent2_vs_fixed_strat_scores)
    scores = torch.stack(scores)
    avg_scores = scores.mean(dim=1)
    avg_vs_fixed_strat_scores = (agent1_vs_fixed_strat_scores + agent2_vs_fixed_strat_scores) / 2.
    avg_vs_fixed_strat_scores = avg_vs_fixed_strat_scores
    avg_vs_alld = avg_vs_fixed_strat_scores[:, 0]
    avg_vs_allc = avg_vs_fixed_strat_scores[:, 1]
    avg_vs_tft = avg_vs_fixed_strat_scores[:, 2]
    if w_coin_record:
        same_colour_coins_record, diff_colour_coins_record = info
        same_colour_coins_record = torch.FloatTensor(
            same_colour_coins_record)
        diff_colour_coins_record = torch.FloatTensor(
            diff_colour_coins_record)
        prop_same_coins = same_colour_coins_record / (
                    same_colour_coins_record + diff_colour_coins_record)
        return avg_scores, avg_vs_alld, avg_vs_allc, avg_vs_tft, prop_same_coins

    return avg_scores, avg_vs_alld, avg_vs_allc, avg_vs_tft, None


def get_scores(ckpts, max_iter_plot=200, w_coin_record=False):
    score_record = []
    avg_vs_alld_record = []
    avg_vs_allc_record = []
    avg_vs_tft_record = []
    if w_coin_record:
        coin_record = []
    for i in range(len(ckpts)):
        ckpts_sublist = ckpts[i]
        if isinstance(ckpts_sublist, list):
            score_subrecord = []
            avg_vs_alld_subrecord = []
            avg_vs_allc_subrecord = []
            avg_vs_tft_subrecord = []
            coin_subrecord = []
            for ckpt in ckpts_sublist:
                full_path = f"{file_path}/{ckpt}"
                avg_scores, avg_vs_alld, avg_vs_allc, avg_vs_tft, prop_same_coins = get_score_individual_ckpt(
                    full_path, w_coin_record=w_coin_record)
                score_subrecord.append(avg_scores)
                avg_vs_alld_subrecord.append(avg_vs_alld)
                avg_vs_allc_subrecord.append(avg_vs_allc)
                avg_vs_tft_subrecord.append(avg_vs_tft)
                coin_subrecord.append(prop_same_coins)
            avg_scores = torch.cat(score_subrecord)[:max_iter_plot]
            avg_vs_alld = torch.cat(avg_vs_alld_subrecord)[:max_iter_plot]
            avg_vs_allc = torch.cat(avg_vs_allc_subrecord)[:max_iter_plot]
            avg_vs_tft = torch.cat(avg_vs_tft_subrecord)[:max_iter_plot]
            if w_coin_record:
                prop_c = torch.cat(coin_subrecord)[:max_iter_plot]

        else:
            full_path = f"{file_path}/{ckpts[i]}"
            avg_scores, avg_vs_alld, avg_vs_allc, avg_vs_tft, prop_c = get_score_individual_ckpt(full_path, w_coin_record=w_coin_record)

        score_record.append(torch.tensor(avg_scores)[:max_iter_plot])
        avg_vs_alld_record.append(avg_vs_alld[:max_iter_plot])
        avg_vs_allc_record.append(avg_vs_allc[:max_iter_plot])
        avg_vs_tft_record.append(avg_vs_tft[:max_iter_plot])
        if w_coin_record:
            coin_record.append(prop_c[:max_iter_plot])

    score_record = torch.stack(score_record)
    avg_vs_alld_record = torch.stack(avg_vs_alld_record)
    avg_vs_allc_record = torch.stack(avg_vs_allc_record)
    avg_vs_tft_record = torch.stack(avg_vs_tft_record)
    if w_coin_record:
        coin_record = torch.stack(coin_record)
        return score_record, avg_vs_alld_record, avg_vs_allc_record, avg_vs_tft_record, coin_record

    return score_record, avg_vs_alld_record, avg_vs_allc_record, avg_vs_tft_record


def plot_coins_record(ckpts, max_iter_plot, label, z_score=1.96, skip_step=0):

    prop_same_coins_record = get_prop_same_coins(ckpts, max_iter_plot=max_iter_plot)

    plot_with_conf_bounds(prop_same_coins_record, max_iter_plot, len(ckpts), label, skip_step,
                          z_score)

def plot_with_conf_bounds(record, max_iter_plot, num_ckpts, label, skip_step, z_score, use_ax=False, ax=None):
    avg = record.mean(dim=0)

    stdev = torch.std(record, dim=0)

    upper_conf_bound = avg + z_score * stdev / np.sqrt(
        num_ckpts)
    lower_conf_bound = avg - z_score * stdev / np.sqrt(
        num_ckpts)

    if use_ax:
        assert ax is not None
        ax.plot(np.arange(max_iter_plot) * skip_step, avg,
             label=label)
        ax.fill_between(np.arange(max_iter_plot) * skip_step, lower_conf_bound,
                     upper_conf_bound, alpha=0.3)

    else:
        plt.plot(np.arange(max_iter_plot) * skip_step, avg,
                 label=label)
        plt.fill_between(np.arange(max_iter_plot) * skip_step, lower_conf_bound,
                         upper_conf_bound, alpha=0.3)


def setup_ipd_plots(titles):
    nfigs = len(titles)
    fig, axs = plt.subplots(1, nfigs, figsize=(5 * (nfigs) + 3, 4))

    for i in range(nfigs):
        axs[i].set_title(titles[i])
        axs[i].set_xlabel("Number of Environment Rollouts")
        axs[i].set_ylabel("Score (Average over Agents and Rollout Length)")

    return fig, axs

def setup_coin_plots(titles):
    nfigs = len(titles)
    fig, axs = plt.subplots(1, nfigs, figsize=(4 * (nfigs) + 6, 4))

    axs[0].set_title(titles[0])
    axs[0].set_xlabel("Number of Environment Rollouts")
    axs[0].set_ylabel("Proportion of Same Coins Picked Up")

    for i in range(1, nfigs):
        axs[i].set_title(titles[i])
        axs[i].set_xlabel("Number of Environment Rollouts")
        axs[i].set_ylabel("Score (Average over Agents and Rollout Length)")

    return fig, axs

def plot_ipd_results(axs, ckpts, nfigs, max_iter_plot, label, z_score=1.96, skip_step=0):

    score_record, avg_vs_alld_record, avg_vs_allc_record, avg_vs_tft_record = \
        get_scores(ckpts, max_iter_plot=max_iter_plot)

    plot_tup = (score_record, avg_vs_alld_record, avg_vs_allc_record, avg_vs_tft_record)

    for i in range(nfigs):
        plot_with_conf_bounds(plot_tup[i], max_iter_plot, len(ckpts), label,
                              skip_step, z_score, use_ax=True, ax=axs[i])
        axs[i].legend()

def plot_coin_results(axs, ckpts, nfigs, max_iter_plot, label, z_score=1.96, skip_step=0):

    score_record, avg_vs_alld_record, avg_vs_allc_record, avg_vs_tft_record, prop_same_coins_record = \
        get_scores(ckpts, max_iter_plot=max_iter_plot, w_coin_record=True)

    plot_tup = (prop_same_coins_record, score_record, avg_vs_alld_record, avg_vs_allc_record, avg_vs_tft_record)

    for i in range(nfigs):
        plot_with_conf_bounds(plot_tup[i], max_iter_plot, len(ckpts), label,
                              skip_step, z_score, use_ax=True, ax=axs[i])
        axs[i].legend()



if plot_coin:
    pola_max_iters = 150
    pola_skip_step = 200
    lola_skip_step = 3
    lola_max_iters = pola_max_iters * pola_skip_step // lola_skip_step

    titles = ("Proportion of Same Coins Picked Up", "Average Score vs Each Other", "Average Score vs Always Defect")
    fig, axs = setup_coin_plots(titles)

    # POLA is 200 skip step because 1 inner 1 outer, 100 times = 200 env rollouts per epoch per agent
    plot_coin_results(axs, ckpts_pola, nfigs=len(titles), max_iter_plot=pola_max_iters, skip_step=pola_skip_step, label="POLA-DiCE")

    # LOLA is 2 skip step because 1 inner 1 outer, 1 time = 2 env rollouts per epoch per agent
    plot_coin_results(axs, ckpts_lola, nfigs=len(titles), max_iter_plot=lola_max_iters, skip_step=lola_skip_step, label="LOLA-DiCE")

    # For OM I'm not counting the env rollouts used for the OM data collection
    plot_coin_results(axs, ckpts_pola_om, nfigs=len(titles), max_iter_plot=pola_max_iters, skip_step=pola_skip_step, label="POLA-OM")

else:
    pola_max_iters = 100
    pola_skip_step = 200
    lola_skip_step = 3
    lola_max_iters = pola_max_iters * pola_skip_step // lola_skip_step

    # titles = ("Average Score vs Each Other", "Average Score vs Always Defect", "Average Score vs Always Cooperate", "Average Score vs TFT")
    titles = ("Average Score vs Each Other", "Average Score vs Always Defect")
    fig, axs = setup_ipd_plots(titles)

    plot_ipd_results(axs, ckpts_pola, nfigs=len(titles), max_iter_plot=pola_max_iters, skip_step=pola_skip_step, label="POLA-DiCE")
    plot_ipd_results(axs, ckpts_lola, nfigs=len(titles), max_iter_plot=lola_max_iters, skip_step=lola_skip_step, label="LOLA-DiCE")
    plot_ipd_results(axs, ckpts_pola_om, nfigs=len(titles), max_iter_plot=pola_max_iters, skip_step=pola_skip_step, label="POLA-OM")
    plot_ipd_results(axs, ckpts_static, nfigs=len(titles), max_iter_plot=2, skip_step=pola_skip_step * 100, label="Random (non-learning)")

    x_vals = np.arange(pola_max_iters) * pola_skip_step
    axs[0].plot(x_vals, -1 * np.ones_like(x_vals), label="Always Cooperate")
    # axs[0].plot(x_vals, -2 * np.ones_like(x_vals), label="Always Defect")
    axs[1].plot(x_vals, -3* np.ones_like(x_vals) , label="Always Cooperate")
    # axs[1].plot(x_vals, -2 * np.ones_like(x_vals), label="Always Defect")
    axs[0].legend()
    axs[1].legend()


plt.show()
