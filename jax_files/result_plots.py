import jax.numpy as jnp
import jax.random
from matplotlib import pyplot as plt

import numpy as np
from POLA_dice_jax import RNN
from flax.training.train_state import TrainState
import optax
from flax.training import checkpoints




load_dir = "."
# load_prefix = f"checkpoint_{now.strftime('%Y-%m-%d_%H-%M')}_seed{args.seed}_epoch"
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

    ckpts_pola = ["checkpoint_2022-09-08_13-36_seed8_epoch251",
                  "checkpoint_2022-09-08_13-34_seed9_epoch251",
                  "checkpoint_2022-09-08_13-49_seed10_epoch251"]

    ckpts_lola = ["checkpoint_2022-09-08_13-36_seed8_epoch251",
                  "checkpoint_2022-09-08_13-34_seed9_epoch251",
                  "checkpoint_2022-09-08_13-49_seed10_epoch251"]


    ckpts_pola_om = ["checkpoint_2022-09-08_13-36_seed8_epoch251",
                  "checkpoint_2022-09-08_13-34_seed9_epoch251",
                  "checkpoint_2022-09-08_13-49_seed10_epoch251"]

else:
    # PLOT IPD

    action_size = 2
    input_size = 6

    ckpts_pola = [
        # "checkpoint_2022-09-08_21-41_seed8_epoch2",
        #           "checkpoint_2022-09-08_21-41_seed8_epoch2"
        # "checkpoint_250_2022-05-05_12-13.pt", # seed 3
        #           "checkpoint_250_2022-05-05_12-16.pt", # seed 4
        #           "checkpoint_200_2022-05-13_00-28_seed15.pt",
                  ]
    ckpts_lola = []
    ckpts_pola_om = []
    ckpts_static = []


def load_from_checkpoint(load_dir, load_prefix, action_size, hidden_size, batch_size, input_size, lr_out, lr_v, outer_optim):
    epoch_num = int(load_prefix.split("epoch")[-1])
    # print(epoch_num)

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

    # score_record = jnp.zeros((epoch_num, 2))
    score_record = [jnp.zeros((2,))] * epoch_num
    vs_fixed_strats_score_record = [[jnp.zeros((3,))] * epoch_num, [jnp.zeros((3,))] * epoch_num]
    # vs_fixed_strats_score_record = [jnp.zeros((epoch_num, 3)), jnp.zeros((epoch_num, 3))]
    same_colour_coins_record = [jnp.zeros((1,))] * epoch_num
    diff_colour_coins_record = [jnp.zeros((1,))] * epoch_num
    coins_collected_info = (
        same_colour_coins_record, diff_colour_coins_record)

    restored_tuple = checkpoints.restore_checkpoint(ckpt_dir=load_dir,
                                                    target=(trainstate_th1, trainstate_val1, trainstate_th2, trainstate_val2,
                                                                coins_collected_info,
                                                                score_record,
                                                                vs_fixed_strats_score_record),
                                                    prefix=load_prefix)

    # restored_tuple = checkpoints.restore_checkpoint(ckpt_dir=load_dir,
    #                                                 target=None,
    #                                                 prefix=load_prefix)

    # print(restored_tuple)
    # print(restored_tuple['4'])
    # print(restored_tuple['6'])
    # 1/0

    trainstate_th1, trainstate_val1, trainstate_th2, trainstate_val2, coins_collected_info, score_record, vs_fixed_strats_score_record = restored_tuple

    same_colour_coins_record, diff_colour_coins_record = coins_collected_info
    same_colour_coins_record = jnp.stack(same_colour_coins_record)
    diff_colour_coins_record = jnp.stack(diff_colour_coins_record)
    coins_collected_info = (same_colour_coins_record, diff_colour_coins_record)

    score_record = jnp.stack(score_record)
    # print(coins_collected_info)
    # print(score_record)
    # print(vs_fixed_strats_score_record)
    vs_fixed_strats_score_record[0] = jnp.stack(vs_fixed_strats_score_record[0])
    vs_fixed_strats_score_record[1] = jnp.stack(vs_fixed_strats_score_record[1])

    # vs_fixed_strats_score_record = jnp.stack(vs_fixed_strats_score_record)
    # print(coins_collected_info)
    # print(score_record)
    # print(vs_fixed_strats_score_record)


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
    # agent1_vs_fixed_strat_scores = jnp.stack(agent1_vs_fixed_strat_scores)
    # agent2_vs_fixed_strat_scores = jnp.stack(agent2_vs_fixed_strat_scores)
    # scores = jnp.stack(score_record)
    avg_scores = score_record.mean(axis=1)

    avg_vs_fixed_strat_scores = (agent1_vs_fixed_strat_scores + agent2_vs_fixed_strat_scores) / 2.
    avg_vs_alld = avg_vs_fixed_strat_scores[:, 0]
    avg_vs_allc = avg_vs_fixed_strat_scores[:, 1]
    avg_vs_tft = avg_vs_fixed_strat_scores[:, 2]
    if w_coin_record:
        same_colour_coins_record, diff_colour_coins_record = coins_collected_info
        # same_colour_coins_record = torch.FloatTensor(
        #     same_colour_coins_record)
        # diff_colour_coins_record = torch.FloatTensor(
        #     diff_colour_coins_record)
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


    print(score_record)
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
        print(i)
        print(plot_tup[i])
        plot_with_conf_bounds(plot_tup[i], max_iter_plot, len(ckpts), label,
                              skip_step, z_score, use_ax=True, ax=axs[i], linestyle=linestyle)
        axs[i].legend()



if plot_coin:
    pola_max_iters = 150
    pola_skip_step = 400
    lola_skip_step = 3
    lola_max_iters = pola_max_iters * pola_skip_step // lola_skip_step

    titles = ("Proportion of Same Coins Picked Up", "Average Score vs Each Other", "Average Score vs Always Defect")
    fig, axs = setup_coin_plots(titles)

    # POLA is 200 skip step because 1 inner 1 outer, 100 times = 200 env rollouts per epoch per agent
    plot_coin_results(axs, ckpts_pola, nfigs=len(titles), max_iter_plot=pola_max_iters, skip_step=pola_skip_step, label="POLA-DiCE", linestyle='dashed')

    # LOLA is 2 skip step because 1 inner 1 outer, 1 time = 2 env rollouts per epoch per agent
    # plot_coin_results(axs, ckpts_lola, nfigs=len(titles), max_iter_plot=lola_max_iters, skip_step=lola_skip_step, label="LOLA-DiCE", linestyle='solid')

    # For OM I'm not counting the env rollouts used for the OM data collection
    # plot_coin_results(axs, ckpts_pola_om, nfigs=len(titles), max_iter_plot=pola_max_iters, skip_step=pola_skip_step, label="POLA-OM", linestyle='dotted')

else:
    pola_max_iters = 100
    pola_skip_step = 200
    lola_skip_step = 3
    lola_max_iters = pola_max_iters * pola_skip_step // lola_skip_step

    # titles = ("Average Score vs Each Other", "Average Score vs Always Defect", "Average Score vs Always Cooperate", "Average Score vs TFT")
    titles = ("Average Score vs Each Other", "Average Score vs Always Defect")
    fig, axs = setup_ipd_plots(titles)

    plot_ipd_results(axs, ckpts_pola, nfigs=len(titles), max_iter_plot=pola_max_iters, skip_step=pola_skip_step, label="POLA-DiCE", linestyle='dashed')
    plot_ipd_results(axs, ckpts_lola, nfigs=len(titles), max_iter_plot=lola_max_iters, skip_step=lola_skip_step, label="LOLA-DiCE", linestyle='solid')
    plot_ipd_results(axs, ckpts_pola_om, nfigs=len(titles), max_iter_plot=pola_max_iters, skip_step=pola_skip_step, label="POLA-OM", linestyle='dotted')
    plot_ipd_results(axs, ckpts_static, nfigs=len(titles), max_iter_plot=2, skip_step=pola_skip_step * 100, label="Random (non-learning)", linestyle='dashdot')

    x_vals = np.arange(pola_max_iters) * pola_skip_step
    axs[0].plot(x_vals, -1 * np.ones_like(x_vals), label="Always Cooperate", linestyle=((0, (3, 1, 1, 1, 1, 1))))
    # axs[0].plot(x_vals, -2 * np.ones_like(x_vals), label="Always Defect")
    axs[1].plot(x_vals, -3* np.ones_like(x_vals) , label="Always Cooperate", linestyle=((0, (3, 1, 1, 1, 1, 1))))
    # axs[1].plot(x_vals, -2 * np.ones_like(x_vals), label="Always Defect")
    axs[0].legend()
    axs[1].legend()


plt.show()

fig.savefig('fig.png')
