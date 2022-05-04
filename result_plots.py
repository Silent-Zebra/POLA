import torch
from LOLA_dice import Agent
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



# TODO LOAD UP AND COMPARE THE LOLA COIN RESULTS. Rerun if needed
# TODO Plot OM results too?!?

ckpts_pola = ["checkpoint_551_2022-04-21_21-04.pt", "checkpoint_551_2022-04-21_20-48.pt",
         "checkpoint_301_2022-04-26_18-06.pt", "checkpoint_301_2022-04-26_18-18.pt",
         "checkpoint_301_2022-04-26_18-22.pt"]
file_path = "."

ckpts_lola = ["checkpoint_20000_2022-04-27_07-35.pt", "checkpoint_20000_2022-04-27_08-28.pt",
              "checkpoint_20000_2022-04-27_07-56.pt", "checkpoint_20000_2022-04-27_07-41.pt"]

ckpts_pola_om = ["checkpoint_201_2022-04-26_16-34.pt", "checkpoint_201_2022-04-26_16-37.pt",
                 "checkpoint_200_2022-04-28_03-55.pt", "checkpoint_200_2022-04-28_04-24.pt",
                 "checkpoint_200_2022-04-28_05-06.pt"]

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


def plot_coins_record(ckpts, max_iter_plot, label, z_score=1.96, skip_step=0):

    prop_same_coins_record = get_prop_same_coins(ckpts, max_iter_plot=max_iter_plot)

    avg_same_coins_total = prop_same_coins_record.mean(dim=0)

    stdev = torch.std(prop_same_coins_record, dim=0)

    upper_conf_bound = avg_same_coins_total + z_score * stdev / np.sqrt(len(ckpts))
    lower_conf_bound = avg_same_coins_total - z_score * stdev / np.sqrt(len(ckpts))

    # print(avg_same_coins_total)
    plt.plot(np.arange(max_iter_plot) * skip_step, avg_same_coins_total, label=label)
    plt.fill_between(np.arange(max_iter_plot) * skip_step, lower_conf_bound, upper_conf_bound, alpha=0.3)
    # plt.fill_between(np.arange(max_iter_plot), lower_conf_bound, upper_conf_bound, edgecolor='b', facecolor='b', alpha=0.3)


plot_coins_record(ckpts_pola, max_iter_plot=200, skip_step=200, label="POLA-DiCE")
# POLA is 200 skip step because 1 inner 1 outer, 100 times = 200 env rollouts per epoch per agent

plot_coins_record(ckpts_lola, max_iter_plot=20000, skip_step=2, label="LOLA-DiCE")
# LOLA is 2 skip step because 1 inner 1 outer, 1 time = 2 env rollouts per epoch per agent

plot_coins_record(ckpts_pola_om, max_iter_plot=200, skip_step=200, label="POLA-OM")
# For OM I guess I'm not counting the env rollouts used for the OM data collection...

plt.legend()
plt.xlabel('Number of Environment Rollouts')
plt.ylabel('Proportion of Same Coins Picked Up')
plt.show()
