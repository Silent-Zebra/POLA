import jax.numpy as jnp
import jax.random
from matplotlib import pyplot as plt

import numpy as np
from POLA_dice_jax import RNN
from flax.training.train_state import TrainState
import optax
from flax.training import checkpoints


load_dir = "."
hidden_size = 64
batch_size = 2000
lr_out = 0.005
lr_v = 0.00005
outer_optim = "adam" # outer optim
layers_before_gru = 1
plot_coin = True

if plot_coin:

    action_size = 4
    input_size = 36

    # Example POLA command:
    # python ./jax_files/POLA_dice_jax.py --env coin  --n_update 302 --gae_lambda 1.0  --inner_steps 2 --outer_steps 200 --lr_in 0.02 --lr_out 0.003 --lr_v 0.0005 --batch_size 2000 --rollout_len 50  --print_every 1 --outer_beta 150 --inner_beta 5 --seed 1 --layers_before_gru 1 --save_dir ./checkpoints/coin/1 --checkpoint_every 25 --hidden_size 64
    ckpts_pola = [
        ["checkpoint_2022-09-29_06-31_seed1_epoch76", "checkpoint_2022-09-29_16-05_seed1_epoch201"],
        ["checkpoint_2022-09-28_20-26_seed2_epoch51", "checkpoint_2022-09-29_16-11_seed2_epoch201"],
        ["checkpoint_2022-09-28_21-41_seed3_epoch26", "checkpoint_2022-09-29_18-07_seed3_epoch226"],
        ["checkpoint_2022-09-13_09-05_seed4_epoch101", "checkpoint_2022-09-13_23-24_seed4_epoch126", "checkpoint_2022-09-19_19-19_seed4_epoch26"],
        ["checkpoint_2022-09-13_09-05_seed5_epoch101", "checkpoint_2022-09-13_23-48_seed5_epoch126", "checkpoint_2022-09-19_21-21_seed5_epoch51"],
        ["checkpoint_2022-09-13_08-58_seed6_epoch101", "checkpoint_2022-09-14_01-20_seed6_epoch151"],
        "checkpoint_2022-09-15_17-44_seed7_epoch326",
        "checkpoint_2022-09-15_19-15_seed8_epoch351",
        "checkpoint_2022-09-15_14-12_seed9_epoch276",
        ["checkpoint_2022-09-28_21-41_seed10_epoch26", "checkpoint_2022-09-29_17-56_seed10_epoch226"]
    ]

    # Example LOLA command:
    # python ./jax_files/POLA_dice_jax.py --env coin  --n_update 50001 --gae_lambda 1.0  --inner_steps 1 --outer_steps 1 --lr_in 0.003 --lr_out 0.003 --lr_v 0.0005 --batch_size 2000 --rollout_len 50  --print_every 10 --outer_beta 0 --inner_beta 0 --seed 1 --layers_before_gru 1 --save_dir ./checkpoints/coin/1 --checkpoint_every 2000 --hidden_size 64
    ckpts_lola = [["checkpoint_2022-10-03_10-34_seed1_epoch30000", "checkpoint_2022-10-03_22-26_seed1_epoch20000"],
                  ["checkpoint_2022-10-03_09-34_seed2_epoch30000", "checkpoint_2022-10-03_22-15_seed2_epoch20000"],
                  ["checkpoint_2022-10-03_10-31_seed3_epoch30000", "checkpoint_2022-10-03_22-03_seed3_epoch20000"],
                  "checkpoint_2022-10-04_06-27_seed4_epoch50000",
                  "checkpoint_2022-10-04_07-19_seed5_epoch50000",
                  "checkpoint_2022-10-04_07-35_seed6_epoch50000",
                  "checkpoint_2022-10-04_06-25_seed7_epoch50000",
                  "checkpoint_2022-10-04_06-40_seed8_epoch50000",
                  "checkpoint_2022-10-04_06-23_seed9_epoch50000",
                  "checkpoint_2022-10-04_06-14_seed10_epoch50000"
                  ]


    # Example POLA OM command:
    # python ./jax_files/POLA_dice_jax.py --env coin  --n_update 302 --gae_lambda 1.0  --inner_steps 4 --outer_steps 200 --lr_in 0.01 --lr_out 0.003 --lr_v 0.0005 --batch_size 1000 --rollout_len 50  --print_every 1 --outer_beta 150 --inner_beta 10 --seed 7 --layers_before_gru 1 --save_dir ./checkpoints/coin/14 --checkpoint_every 25 --hidden_size 64 --opp_model --opp_model_steps 1 --opp_model_data_batches 200 --om_lr_p 0.005 --om_lr_v 0.0005
    ckpts_pola_om = [
        "checkpoint_2022-09-20_09-31_seed1_epoch251",
        "checkpoint_2022-09-20_10-19_seed2_epoch251",
        "checkpoint_2022-09-30_11-25_seed3_epoch251",
        "checkpoint_2022-09-30_07-43_seed4_epoch251",
        ["checkpoint_2022-09-28_22-44_seed5_epoch51", "checkpoint_2022-09-30_12-44_seed5_epoch181", "checkpoint_2022-10-01_05-36_seed5_epoch51"],
        "checkpoint_2022-09-20_09-22_seed6_epoch251",
        ["checkpoint_2022-09-18_17-11_seed7_epoch201", "checkpoint_2022-09-19_12-11_seed7_epoch76"],
        ["checkpoint_2022-09-18_16-44_seed8_epoch201", "checkpoint_2022-09-19_12-07_seed8_epoch76"],
        ["checkpoint_2022-09-18_16-51_seed9_epoch226", "checkpoint_2022-09-19_21-44_seed9_epoch151"],
        "checkpoint_2022-09-30_07-34_seed10_epoch251"
        ]

else:
    # PLOT IPD

    action_size = 2
    input_size = 6

    # python ./jax_files/POLA_dice_jax.py --env ipd  --n_update 151 --gae_lambda 1.0  --inner_steps 2 --outer_steps 200 --lr_in 0.005 --lr_out 0.003 --lr_v 0.0005 --batch_size 2000 --rollout_len 50  --print_every 1 --inner_beta 10 --outer_beta 100 --seed 1 --layers_before_gru 1 --save_dir ./checkpoints/ipd/21 --checkpoint_every 10 --hidden_size 64
    ckpts_pola = [
        "checkpoint_2022-10-06_12-06_seed1_epoch150",
        "checkpoint_2022-10-06_11-50_seed2_epoch150",
        "checkpoint_2022-10-06_12-12_seed3_epoch150",
        "checkpoint_2022-10-06_12-04_seed4_epoch150",
        "checkpoint_2022-10-06_11-48_seed5_epoch150",
        "checkpoint_2022-10-06_11-56_seed6_epoch150",
        "checkpoint_2022-10-06_12-12_seed7_epoch150",
        "checkpoint_2022-10-06_12-04_seed8_epoch150",
        "checkpoint_2022-10-06_12-09_seed9_epoch150",
        "checkpoint_2022-10-06_12-07_seed10_epoch150"
    ]

    # python ./jax_files/POLA_dice_jax.py --env ipd  --n_update 20001 --gae_lambda 1.0  --inner_steps 1 --outer_steps 1 --lr_in 0.05 --lr_out 0.003 --lr_v 0.0005 --batch_size 2000 --rollout_len 50  --print_every 10 --outer_beta 0 --inner_beta 0 --seed 6 --layers_before_gru 1 --save_dir ./checkpoints/ipd/51 --checkpoint_every 1000 --hidden_size 64 --contrib_factor 1.33
    ckpts_lola = [
        "checkpoint_2022-10-05_08-21_seed1_epoch20000",
        "checkpoint_2022-10-05_08-01_seed2_epoch20000",
        "checkpoint_2022-10-05_07-59_seed3_epoch20000",
        "checkpoint_2022-10-05_08-04_seed4_epoch20000",
        "checkpoint_2022-10-05_08-01_seed5_epoch20000"
        "checkpoint_2022-10-05_18-52_seed6_epoch20000",
        "checkpoint_2022-10-05_18-49_seed7_epoch20000",
        "checkpoint_2022-10-05_18-48_seed8_epoch20000",
        "checkpoint_2022-10-05_18-43_seed9_epoch20000",
        "checkpoint_2022-10-05_18-50_seed10_epoch20000"
    ]


    # python ./jax_files/POLA_dice_jax.py --env ipd  --n_update 151 --gae_lambda 1.0  --inner_steps 2 --outer_steps 200 --lr_in 0.005 --lr_out 0.003 --lr_v 0.0005 --batch_size 2000 --rollout_len 50  --print_every 1 --inner_beta 10 --outer_beta 100 --seed 10 --layers_before_gru 1 --save_dir ./checkpoints/ipd/10 --checkpoint_every 10 --hidden_size 64 --opp_model --opp_model_steps 1 --opp_model_data_batches 200 --om_lr_p 0.005 --om_lr_v 0.0005
    ckpts_pola_om = [
        "checkpoint_2022-10-06_13-08_seed1_epoch150",
        "checkpoint_2022-10-06_13-48_seed2_epoch150",
        "checkpoint_2022-10-06_13-29_seed3_epoch150",
        "checkpoint_2022-10-06_13-27_seed4_epoch150",
        "checkpoint_2022-10-06_13-28_seed5_epoch150",
        "checkpoint_2022-10-06_13-28_seed6_epoch150",
        "checkpoint_2022-10-06_13-26_seed7_epoch150",
        "checkpoint_2022-10-06_13-29_seed8_epoch150",
        "checkpoint_2022-10-06_13-27_seed9_epoch150",
        "checkpoint_2022-10-06_13-27_seed10_epoch150"
    ]

    ckpts_static = []


def load_from_checkpoint(load_dir, load_prefix, action_size, hidden_size, batch_size, input_size, lr_out, lr_v, outer_optim):
    epoch_num = int(load_prefix.split("epoch")[-1])

    if epoch_num % 10 == 0:
        epoch_num += 1 # Kind of an ugly temporary fix to allow for the updated checkpointing system which now has
        # record of rewards/eval vs fixed strat before the first training - important for IPD plots. Should really be applied to
        # all checkpoints with the new updated code I have, but the coin checkpoints above are from old code

    theta_p1 = RNN(num_outputs=action_size,
                   num_hidden_units=hidden_size, layers_before_gru=layers_before_gru)
    theta_v1 = RNN(num_outputs=1, num_hidden_units=hidden_size, layers_before_gru=layers_before_gru)

    theta_p1_params = theta_p1.init(jax.random.PRNGKey(0), jnp.ones(
        [batch_size, input_size]), jnp.zeros(hidden_size))
    theta_v1_params = theta_v1.init(jax.random.PRNGKey(0), jnp.ones(
        [batch_size, input_size]), jnp.zeros(hidden_size))

    theta_p2 = RNN(num_outputs=action_size,
                   num_hidden_units=hidden_size, layers_before_gru=layers_before_gru)
    theta_v2 = RNN(num_outputs=1, num_hidden_units=hidden_size, layers_before_gru=layers_before_gru)

    theta_p2_params = theta_p2.init(jax.random.PRNGKey(0), jnp.ones(
        [batch_size, input_size]), jnp.zeros(hidden_size))
    theta_v2_params = theta_v2.init(jax.random.PRNGKey(0), jnp.ones(
        [batch_size, input_size]), jnp.zeros(hidden_size))

    if outer_optim.lower() == 'adam':
        theta_optimizer = optax.adam(learning_rate=lr_out)
        value_optimizer = optax.adam(learning_rate=lr_v)
    elif outer_optim.lower() == 'sgd':
        theta_optimizer = optax.sgd(learning_rate=lr_out)
        value_optimizer = optax.sgd(learning_rate=lr_v)
    else:
        raise Exception("Unknown or Not Implemented Optimizer")

    trainstate_th1 = TrainState.create(apply_fn=theta_p1.apply,
                                       params=theta_p1_params,
                                       tx=theta_optimizer)
    trainstate_val1 = TrainState.create(apply_fn=theta_v1.apply,
                                        params=theta_v1_params,
                                        tx=value_optimizer)
    trainstate_th2 = TrainState.create(apply_fn=theta_p2.apply,
                                       params=theta_p2_params,
                                       tx=theta_optimizer)
    trainstate_val2 = TrainState.create(apply_fn=theta_v2.apply,
                                        params=theta_v2_params,
                                        tx=value_optimizer)

    score_record = [jnp.zeros((2,))] * epoch_num
    vs_fixed_strats_score_record = [[jnp.zeros((3,))] * epoch_num, [jnp.zeros((3,))] * epoch_num]
    if plot_coin:

        same_colour_coins_record = [jnp.zeros((1,))] * epoch_num
        diff_colour_coins_record = [jnp.zeros((1,))] * epoch_num
    else:
        same_colour_coins_record = []
        diff_colour_coins_record = []
    coins_collected_info = (
        same_colour_coins_record, diff_colour_coins_record)

    restored_tuple = checkpoints.restore_checkpoint(ckpt_dir=load_dir,
                                                    target=(trainstate_th1, trainstate_val1, trainstate_th2, trainstate_val2,
                                                                coins_collected_info,
                                                                score_record,
                                                                vs_fixed_strats_score_record),
                                                    prefix=load_prefix)


    trainstate_th1, trainstate_val1, trainstate_th2, trainstate_val2, coins_collected_info, score_record, vs_fixed_strats_score_record = restored_tuple

    if plot_coin:
        same_colour_coins_record, diff_colour_coins_record = coins_collected_info
        same_colour_coins_record = jnp.stack(same_colour_coins_record)
        diff_colour_coins_record = jnp.stack(diff_colour_coins_record)
        coins_collected_info = (same_colour_coins_record, diff_colour_coins_record)

    score_record = jnp.stack(score_record)

    vs_fixed_strats_score_record[0] = jnp.stack(vs_fixed_strats_score_record[0])
    vs_fixed_strats_score_record[1] = jnp.stack(vs_fixed_strats_score_record[1])



    return score_record, vs_fixed_strats_score_record,  coins_collected_info



def get_prop_same_coins(ckpts, max_iter_plot=200):
    prop_same_coins_record = []
    for i in range(len(ckpts)):
        load_prefix = ckpts[i]
        score_record, vs_fixed_strats_score_record, coins_collected_info = load_from_checkpoint(load_dir, load_prefix, action_size, hidden_size, batch_size, input_size, lr_out, lr_v, outer_optim)
        same_colour_coins_record, diff_colour_coins_record = coins_collected_info
        same_colour_coins_record = (same_colour_coins_record[:max_iter_plot])
        diff_colour_coins_record = (diff_colour_coins_record[:max_iter_plot])
        prop_same_coins = same_colour_coins_record / (same_colour_coins_record + diff_colour_coins_record)
        prop_same_coins_record.append(prop_same_coins)
    return jnp.stack(prop_same_coins_record)


def get_score_individual_ckpt(load_dir, load_prefix, w_coin_record=False):
    score_record, vs_fixed_strats_score_record, coins_collected_info = load_from_checkpoint(load_dir, load_prefix, action_size, hidden_size, batch_size, input_size, lr_out, lr_v, outer_optim)

    agent1_vs_fixed_strat_scores, agent2_vs_fixed_strat_scores = vs_fixed_strats_score_record

    avg_scores = score_record.mean(axis=1)

    avg_vs_fixed_strat_scores = (agent1_vs_fixed_strat_scores + agent2_vs_fixed_strat_scores) / 2.
    avg_vs_alld = avg_vs_fixed_strat_scores[:, 0]
    avg_vs_allc = avg_vs_fixed_strat_scores[:, 1]
    avg_vs_tft = avg_vs_fixed_strat_scores[:, 2]
    if w_coin_record:
        same_colour_coins_record, diff_colour_coins_record = coins_collected_info
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
                load_prefix = ckpt
                avg_scores, avg_vs_alld, avg_vs_allc, avg_vs_tft, prop_same_coins = get_score_individual_ckpt(
                    load_dir, load_prefix, w_coin_record=w_coin_record)
                score_subrecord.append(avg_scores)
                avg_vs_alld_subrecord.append(avg_vs_alld)
                avg_vs_allc_subrecord.append(avg_vs_allc)
                avg_vs_tft_subrecord.append(avg_vs_tft)
                coin_subrecord.append(prop_same_coins)
            avg_scores = jnp.concatenate(score_subrecord)[:max_iter_plot]
            avg_vs_alld = jnp.concatenate(avg_vs_alld_subrecord)[:max_iter_plot]
            avg_vs_allc = jnp.concatenate(avg_vs_allc_subrecord)[:max_iter_plot]
            avg_vs_tft = jnp.concatenate(avg_vs_tft_subrecord)[:max_iter_plot]
            if w_coin_record:
                prop_c = jnp.concatenate(coin_subrecord)[:max_iter_plot]

        else:
            load_prefix = ckpts[i]
            avg_scores, avg_vs_alld, avg_vs_allc, avg_vs_tft, prop_c = get_score_individual_ckpt(load_dir, load_prefix, w_coin_record=w_coin_record)

        score_record.append(avg_scores[:max_iter_plot])
        avg_vs_alld_record.append(avg_vs_alld[:max_iter_plot])
        avg_vs_allc_record.append(avg_vs_allc[:max_iter_plot])
        avg_vs_tft_record.append(avg_vs_tft[:max_iter_plot])
        if w_coin_record:
            coin_record.append(prop_c[:max_iter_plot])


    score_record = jnp.stack(score_record)
    avg_vs_alld_record = jnp.stack(avg_vs_alld_record)
    avg_vs_allc_record = jnp.stack(avg_vs_allc_record)
    avg_vs_tft_record = jnp.stack(avg_vs_tft_record)
    if w_coin_record:
        coin_record = jnp.stack(coin_record)
        return score_record, avg_vs_alld_record, avg_vs_allc_record, avg_vs_tft_record, coin_record

    return score_record, avg_vs_alld_record, avg_vs_allc_record, avg_vs_tft_record


def plot_coins_record(ckpts, max_iter_plot, label, z_score=1.96, skip_step=0):

    prop_same_coins_record = get_prop_same_coins(ckpts, max_iter_plot=max_iter_plot)

    plot_with_conf_bounds(prop_same_coins_record, max_iter_plot, len(ckpts), label, skip_step,
                          z_score)

def plot_with_conf_bounds(record, max_iter_plot, num_ckpts, label, skip_step, z_score, use_ax=False, ax=None, linestyle='solid'):
    avg = record.mean(axis=0)

    stdev = jnp.std(record, axis=0)

    upper_conf_bound = avg + z_score * stdev / np.sqrt(
        num_ckpts)
    lower_conf_bound = avg - z_score * stdev / np.sqrt(
        num_ckpts)

    if use_ax:
        assert ax is not None
        ax.plot(np.arange(max_iter_plot) * skip_step, avg,
             label=label, linestyle=linestyle)
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
        axs[i].set_xlabel("Total Number of Outer Steps")
        axs[i].set_ylabel("Score (Average over Agents and Rollout Length)")

    return fig, axs

def setup_coin_plots(titles):
    nfigs = len(titles)
    fig, axs = plt.subplots(1, nfigs, figsize=(4 * (nfigs) + 6, 4))

    axs[0].set_title(titles[0])
    axs[0].set_xlabel("Total Number of Outer Steps")
    axs[0].set_ylabel("Proportion of Same Coins Picked Up")

    for i in range(1, nfigs):
        axs[i].set_title(titles[i])
        axs[i].set_xlabel("Total Number of Outer Steps")
        axs[i].set_ylabel("Score (Average over Agents and Rollout Length)")

    return fig, axs

def plot_ipd_results(axs, ckpts, nfigs, max_iter_plot, label, z_score=1.96, skip_step=0, linestyle='solid'):

    score_record, avg_vs_alld_record, avg_vs_allc_record, avg_vs_tft_record = \
        get_scores(ckpts, max_iter_plot=max_iter_plot)

    plot_tup = (score_record, avg_vs_alld_record, avg_vs_allc_record, avg_vs_tft_record)

    for i in range(nfigs):
        plot_with_conf_bounds(plot_tup[i], max_iter_plot, len(ckpts), label,
                              skip_step, z_score, use_ax=True, ax=axs[i], linestyle=linestyle)
        axs[i].legend()

def plot_coin_results(axs, ckpts, nfigs, max_iter_plot, label, z_score=1.96, skip_step=0, linestyle='solid'):

    score_record, avg_vs_alld_record, avg_vs_allc_record, avg_vs_tft_record, prop_same_coins_record = \
        get_scores(ckpts, max_iter_plot=max_iter_plot, w_coin_record=True)

    plot_tup = (prop_same_coins_record, score_record, avg_vs_alld_record, avg_vs_allc_record, avg_vs_tft_record)

    for i in range(nfigs):
        plot_with_conf_bounds(plot_tup[i], max_iter_plot, len(ckpts), label,
                              skip_step, z_score, use_ax=True, ax=axs[i], linestyle=linestyle)
        axs[i].legend()



if plot_coin:
    pola_max_iters = 250 # epochs/n_update
    pola_skip_step = 200 # outer steps
    pola_om_max_iters = pola_max_iters
    pola_om_skip_step = pola_skip_step
    lola_skip_step = 1 # outer steps
    lola_max_iters = pola_max_iters * pola_skip_step // lola_skip_step

    # titles = ("Proportion of Same Coins Picked Up", "Average Score vs Each Other", "Average Score vs Always Defect", "Average Score vs Always Cooperate", "Average Score vs TFT")
    titles = ("Proportion of Same Coins Picked Up", "Average Score vs Each Other", "Average Score vs Always Defect")
    fig, axs = setup_coin_plots(titles)

    # POLA is 200 skip step because 1 inner 1 outer, 100 times = 200 env rollouts per epoch per agent
    plot_coin_results(axs, ckpts_pola, nfigs=len(titles), max_iter_plot=pola_max_iters, skip_step=pola_skip_step, label="POLA-DiCE", linestyle='dashed')

    # LOLA is 2 skip step because 1 inner 1 outer, 1 time = 2 env rollouts per epoch per agent
    plot_coin_results(axs, ckpts_lola, nfigs=len(titles), max_iter_plot=lola_max_iters, skip_step=lola_skip_step, label="LOLA-DiCE", linestyle='solid')

    # For OM I'm not counting the env rollouts used for the OM data collection
    plot_coin_results(axs, ckpts_pola_om, nfigs=len(titles), max_iter_plot=pola_om_max_iters, skip_step=pola_om_skip_step, label="POLA-OM", linestyle='dotted')


    # Agents who always cooperate only pick up their own coins, and a coin is picked up on average every 1.5 time steps, so 3 time steps for each agent to pick up a coin, so the maximum expected reward is 1/3 per time step. If you always cooperate against an always defect agent, a coin is picked up on average every 1.5 time steps, but the cooperative agent gets 0 reward half of the time, and for the remaining 25 time steps, the coop agent competes with the always defect agent. If both pick up coin, coop agent gets -1, if coop wins race, coop gets 1, if coop loses coop gets -2. 50% of the time, equal distance, so coop gets -1, then 50% of the time it's a race with expected value -0.5. So average -0.75. So 25 time steps every 1.5 steps you get -0.75. So on 16.666 coins you get -0.75 which is 12.5, then /50 you get -0.25 average. Empirically I find it is -0.26. I think the reasoning is not exactly correct because of the possibility of 2 agents being on the same space.
    x_vals = np.arange(pola_max_iters) * pola_skip_step
    axs[0].plot(x_vals, 1. * np.ones_like(x_vals), label="Always Cooperate",
                linestyle=((0, (3, 1, 1, 1, 1, 1))))
    axs[0].plot(x_vals, 0.5 * np.ones_like(x_vals), label="Always Defect",
                linestyle='dashdot')
    axs[1].plot(x_vals, 1. / 3. * np.ones_like(x_vals), label="Always Cooperate",
                linestyle=((0, (3, 1, 1, 1, 1, 1))))
    axs[1].plot(x_vals, 0 * np.ones_like(x_vals), label="Always Defect",
                linestyle='dashdot')
    axs[2].plot(x_vals, -0.26 * np.ones_like(x_vals), label="Always Cooperate",
                linestyle=((0, (3, 1, 1, 1, 1, 1))))
    axs[2].plot(x_vals, 0 * np.ones_like(x_vals), label="Always Defect",
                linestyle='dashdot')
    axs[0].legend()
    axs[1].legend()
    axs[2].legend()

else:
    pola_max_iters = 100 # epochs/n_update
    pola_skip_step = 200 # outer steps
    lola_skip_step = 1 # outer steps
    lola_max_iters = pola_max_iters * pola_skip_step // lola_skip_step

    # titles = ("Average Score vs Each Other", "Average Score vs Always Defect", "Average Score vs Always Cooperate", "Average Score vs TFT")
    titles = ("Average Score vs Each Other", "Average Score vs Always Defect")
    fig, axs = setup_ipd_plots(titles)

    plot_ipd_results(axs, ckpts_pola, nfigs=len(titles), max_iter_plot=pola_max_iters, skip_step=pola_skip_step, label="POLA-DiCE", linestyle='dashed')
    plot_ipd_results(axs, ckpts_lola, nfigs=len(titles), max_iter_plot=lola_max_iters, skip_step=lola_skip_step, label="LOLA-DiCE", linestyle='solid')
    plot_ipd_results(axs, ckpts_pola_om, nfigs=len(titles), max_iter_plot=pola_max_iters, skip_step=pola_skip_step, label="POLA-OM", linestyle='dotted')

    x_vals = np.arange(pola_max_iters) * pola_skip_step
    axs[0].plot(x_vals, 0.33 * np.ones_like(x_vals), label="Always Cooperate", linestyle=((0, (3, 1, 1, 1, 1, 1))))
    axs[0].plot(x_vals, 0 * np.ones_like(x_vals), label="Always Defect", linestyle='dashdot')
    axs[1].plot(x_vals, -0.335 * np.ones_like(x_vals) , label="Always Cooperate", linestyle=((0, (3, 1, 1, 1, 1, 1))))
    axs[1].plot(x_vals, 0 * np.ones_like(x_vals), label="Always Defect", linestyle='dashdot')
    axs[0].legend()
    axs[1].legend()


plt.show()

fig.savefig('fig.png')
