# Some parts are loosely based on https://github.com/aletcher/stable-opponent-shaping/blob/master/stable_opponent_shaping.ipynb

# NOTE: there is a lot of stuff that is unused in this file. Sorry for the loss in
# readability due to that. I tried a bunch of things and it turned out that
# most of them don't work well (either in performance or runtime),
# or are too confusing or off-topic to be helpful.
# Outer POLA is actually not that complicated. A lot of this is for the IFT version of
# POLA which ended up not making it into the final version of the paper.

import numpy as np
import torch
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F

import higher # There is a way to do it without this even with neural nets (e.g. see the LOLA_dice file), but anyway

import datetime

import copy

import argparse

import random

from timeit import default_timer as timer


def bin_inttensor_from_int(x, n):
    """Converts decimal value integer x into binary representation.
    Parameter n represents the number of agents (so you fill with 0s up to the number of agents)
    Well n doesn't have to be num agents. In case of lookback (say 2 steps)
    then we may want n = 2x number of agents"""
    return torch.Tensor([int(d) for d in (str(bin(x))[2:]).zfill(n)])


def build_bin_matrix(n, size):
    bin_mat = torch.zeros((size, n))
    for i in range(size):
        l = bin_inttensor_from_int(i, n)
        bin_mat[i] = l
    return bin_mat


def build_p_vector(n, size, pc, bin_mat):
    pc = pc.repeat(size).reshape(size, n)
    pd = 1 - pc
    p = torch.prod(bin_mat * pc + (1 - bin_mat) * pd, dim=1)
    return p


def copyNN(copy_to_net, copy_from_net):
    copy_to_net.load_state_dict(copy_from_net.state_dict())


def optim_update(optim, loss, params=None):
    if params is not None:
        # diffopt step here
        return optim.step(loss, params)
    else:
        optim.zero_grad()
        loss.backward(retain_graph=True)
        optim.step()


class Game():
    def __init__(self, n, init_state_representation, history_len=1):
        self.n_agents = n
        self.history_len = history_len
        self.init_state_representation = init_state_representation
        if args.ill_condition:
            self.dd_stretch_factor = args.dd_stretch_factor  # 30
            self.all_state_stretch_factor = args.all_state_stretch_factor  # 0.1
            # So then dd_stretch_factor * all state stretch is what you get in the DD state

    def print_policy_info(self, policy, i):
        print("Policy {}".format(i + 1), flush=True)
        # print("(Probabilities are for cooperation/contribution, for states 00...0 (no contrib,..., no contrib), 00...01 (only last player contrib), 00...010, 00...011, increasing in binary order ..., 11...11 , start)")
        print(policy)

    def print_reward_info(self, G_ts, discounted_sum_of_adjustments,
                          truncated_coop_payout, inf_coop_payout, env):

        print(
            "Discounted Sum Rewards (Avg over batches) in this episode (removing negative adjustment): ")
        print(G_ts[0].mean(dim=1).reshape(-1) + discounted_sum_of_adjustments)

        if env == 'ipd':
            print("Max Avg Coop Payout (Truncated Horizon): {:.3f}".format(
                truncated_coop_payout))
            print("Max Avg Coop Payout (Infinite Horizon): {:.3f}".format(
                inf_coop_payout))

    def build_all_combs_state_batch(self):

        dim = self.n_agents * self.history_len

        state_batch = torch.cat((build_bin_matrix(dim, 2 ** dim),
                                 torch.Tensor(
                                     [init_state_representation] * dim).reshape(
                                     1, -1)))

        state_batch = self.build_one_hot_from_batch(state_batch.t(),
                                                    self.action_repr_dim,
                                                    one_at_a_time=False)

        return state_batch

    def get_nn_policy_for_batch(self, pol, state_batch):
        if args.ill_condition:
            simple_state_repr_batch = self.one_hot_to_simple_repr(state_batch)

            simple_mask = (simple_state_repr_batch.sum(dim=-1) == 0).unsqueeze(
                -1)  # DD state

            policy = torch.sigmoid(
                pol(state_batch) * (self.all_state_stretch_factor) * (
                        (self.dd_stretch_factor - 1) * simple_mask + 1))
            # quite ugly but what this simple_mask does is multiply by (dd stretch factor) in the state DD, and 1 elsewhere
            # when combined with the all_state_stretch_factor, the effect is to magnify the DD state updates (policy amplified away from 0.5),
            # and scale down the updates in other states (policy brought closer to 0.5)
        else:
            policy = torch.sigmoid(pol(state_batch))

        return policy

    def get_policy_for_all_states(self, th, i, ill_cond=False):
        if isinstance(th[i], torch.Tensor):
            # if args.ill_condition:
            if ill_cond:
                policy = torch.sigmoid(ill_cond_matrices[i] @ th[i])
            else:
                policy = torch.sigmoid(th[i])

        else:
            state_batch = self.build_all_combs_state_batch()
            policy = self.get_nn_policy_for_batch(th[i], state_batch)
            policy = policy.squeeze(-1)

        return policy

    def learn_om_from_policy(self, th, opp_models, i, j):
        # i is the index for the learning agent
        # j is in the index for the opponent whose policy we have access to
        # and agent i is learning the opponent model of
        # Modifies in place opp_models (which is a list of lists, where each sublist
        # is the set of OMs an agent has of all the other agents)
        agent1oms = opp_models[i]

        # print(agent1oms)
        # game.print_policies_for_all_states(agent1oms)

        agent1om_of_agent2_policy = self.get_policy_for_all_states(agent1oms, j, ill_cond=args.om_precond)
        actual_agent2_policy = self.get_policy_for_all_states(th, j, ill_cond=args.ill_condition)
        policy, target_policy = agent1om_of_agent2_policy, actual_agent2_policy

        kl_div = get_kl_div_from_policies(policy, target_policy, i,
                                          policies_are_logits=False)

        # print("OM")
        # print(agent1om_of_agent2_policy)
        # print("Real")
        # print(actual_agent2_policy)

        with torch.no_grad():
            if isinstance(agent1oms[j], NeuralNet):
                for param in agent1oms[j].parameters():
                    param_grad = get_gradient(kl_div, param)
                    param.data -= args.om_lr_p * param_grad

            else:
                agent1oms[j] -= args.om_lr_p * get_gradient(kl_div,
                                                            agent1oms[j])

        # print(kl_div)
        # game.print_policies_for_all_states(agent1oms)
        return kl_div

    def print_policies_for_all_states(self, th, ill_cond=False):
        for i in range(len(th)):
            policy = self.get_policy_for_all_states(th, i, ill_cond=ill_cond)
            self.print_policy_info(policy, i)

    def build_one_hot_from_batch(self, curr_step_batch, one_hot_dim,
                                 one_at_a_time=True, range_end=None,
                                 simple_2state_build=False):

        if range_end is None:
            range_end = self.n_agents
        curr_step_batch_one_hot = torch.nn.functional.one_hot(
            curr_step_batch.long(), one_hot_dim).squeeze(dim=2)

        if simple_2state_build:
            new_tens = torch.cat((curr_step_batch_one_hot[:, 0, :],
                                  curr_step_batch_one_hot[:, 1, :]), dim=-1)
        else:
            new_tens = curr_step_batch_one_hot[0]
            if not one_at_a_time:
                range_end *= self.history_len

            for i in range(1, range_end):
                new_tens = torch.cat((new_tens, curr_step_batch_one_hot[i]),
                                     dim=-1)

        curr_step_batch = new_tens.float()
        return curr_step_batch


class ContributionGame(Game):
    """
    The way this is structured, 1 means a contribution of 1 (and is therefore cooperation) and 0 is a contribution of 0, which is defecting
    The game works conceptually as: at each round, an agent can either contribute 0 or 1.
    The total number of contributions go into a public pool (e.g. consider some investment fund, or investing in infrastructure, or something along those lines)
    which is redistributed equally to agents (all agents benefit equally from the public investment/infrastructure).
    The value each agent gets from investing 1 must be <1 for the agent itself, but the total value (if you sum up the value each individual agent has across all agents)
    must be > 1. (So if contribution_scale = False, contribution_factor needs to be > 1, otherwise nobody should ever contribute)
    Contributing 1 provides an individual reward of -1 (in addition to the redistribution of total contributions)
    Contributing 0 provides no individual reward (beyond that from the redistribution of total contributions)
    """

    def __init__(self, n, batch_size, num_iters, gamma=0.96,
                 contribution_factor=1.6,
                 contribution_scale=False, history_len=1):

        super().__init__(n,
                         init_state_representation=args.init_state_representation,
                         history_len=history_len)

        self.gamma = gamma
        self.contribution_factor = contribution_factor
        self.contribution_scale = contribution_scale
        self.batch_size = batch_size
        self.num_iters = num_iters

        self.action_repr_dim = 3  # one hot with 3 dimensions, dimension 0 for defect, 1 for contrib/coop, 2 for start

        if args.using_nn:
            self.dims = [n * history_len * self.action_repr_dim] * n
        else:
            self.dims = [2 ** n + 1] * n
        if args.opp_model:
            if args.om_using_nn:
                self.om_dims = [n * history_len * self.action_repr_dim] * n
            else:
                self.om_dims = [2 ** n + 1] * n

        """
        for dims, the last n is the number of agents, basically dims[i] is the dim for each agent
        It's sort of a silly way to set things up in the event that all agents are the same
        which is what I am currently doing for all of my experiments
        but would make sense if you mix agents with different state representations
        But I am not sure why you would want to mix the agents like that (giving
        different agents different vision/observations of the same underlying state, essentially)
        """

        if self.contribution_scale:
            self.contribution_factor = contribution_factor * n
        # else:
        #     assert self.contribution_factor > 1

        self.dec_value_mask = (2 ** torch.arange(n - 1, -1, -1)).float()

        # For exact calculations
        self.state_space = self.dims[0]
        self.bin_mat = build_bin_matrix(self.n_agents, 2 ** self.n_agents)
        self.payout_vectors = torch.zeros((n, 2 ** self.n_agents))  # one vector for each player, state space - 1 because one is the initial state. This is the r^1 or r^2 in the LOLA paper exact gradient formulation. In the 2p case this is for DD, DC, CD, CC

        for agent in range(n):
            for state in range(2 ** self.n_agents):
                l = bin_inttensor_from_int(state, n)
                total_contrib = sum(l)
                agent_payout = total_contrib * contribution_factor / n - l[
                    agent]  # if agent contributed 1, subtract 1
                agent_payout -= adjustment_to_make_rewards_negative
                self.payout_vectors[agent][state] = agent_payout


    def get_exact_loss(self, th, return_p_mat_only=False, ill_cond=False, self_no_cond=False, self_index=-1 ):
        """
        Theta denotes (unnormalized) action probabilities at each of the states:
        DD DC CD CC start (where DD = both defect, DC = agent 1 defects while agent 2 cooperates, etc.)
        Note that this is different from the original LOLA/SOS formulation which uses start CC CD DC DD
        The reason I flipped this is because, when thinking about the contribution game which is the generalization of the IPD,
        It is convenient to think of 0 as no contribution (i.e. defect) and 1 as contribute (i.e. cooperate)
        And using binary construction from 00, 01, 10, 11 is easy to work with (counting up)
        In the 3p case, we go from 000, 001, ..., 111
        In the n-player case, 000...000 is all defect, 111...111 is all cooperate.
        """

        if self_no_cond:
            assert self_index >= 0


        init_pc = torch.zeros(self.n_agents)

        policies = []
        for i in range(self.n_agents):

            if self_no_cond and i == self_index:
                policy = self.get_policy_for_all_states(th, i,
                                                        ill_cond=False)
            else:
                policy = self.get_policy_for_all_states(th, i, ill_cond=ill_cond)
            policies.append(policy)

            if init_state_representation == 1:
                p_i_0 = policy[-2]  # force all coop at the beginning in this special case
            else:
                p_i_0 = policy[-1]  # start state is at the end; this is a (kind of arbitrary) design choice

            init_pc[i] = p_i_0
            # Policy represents prob of coop (taking action 1)

        p = build_p_vector(self.n_agents, 2 ** self.n_agents, init_pc,
                           self.bin_mat)

        # This part and the below part might be optimizable (without a loop) but it doesn't seem trivial
        # Probabilities in the states other than the start state
        all_p_is = torch.zeros((self.n_agents, 2 ** self.n_agents))
        for i in range(self.n_agents):
            p_i = policies[i][0:-1]
            all_p_is[i] = p_i.flatten()

        # Transition Matrix
        # Remember now our transition matrix top left is DDD...D
        # 0 is defect in this formulation, 1 is contributing/cooperating
        P = torch.zeros((2 ** self.n_agents, 2 ** self.n_agents))
        for curr_state in range(2 ** self.n_agents):
            i = curr_state
            pc = all_p_is[:, i]
            p_new = build_p_vector(self.n_agents, 2 ** self.n_agents, pc,
                                   self.bin_mat)
            P[i] = p_new

        # Here instead of using infinite horizon which is the implicit assumption in the derivation from the previous parts (see again the LOLA appendix A.2)
        # We can consider a finite horizon and literally just unroll the calculation
        # You could probably just do 2 inverses and do some multiplication by discount factor
        # and subtract instead of doing this loop but oh well, this is more for testing and comparing anyway.
        if args.exact_finite_horizon:
            gamma_t_P_t = torch.eye(2 ** self.n_agents)
            running_total = torch.eye(2 ** self.n_agents)

            for t in range(1, args.rollout_len):
                gamma_t_P_t = gamma * gamma_t_P_t @ P
                running_total += gamma_t_P_t
            M = p @ gamma_t_P_t
        else:
            M = torch.matmul(p, torch.inverse(
                torch.eye(2 ** self.n_agents) - gamma * P))

        # Remember M is just the steady state probabilities for each of the states (discounted state visitation count, not a probability)
        # It is a vector, not a matrix.

        if return_p_mat_only:
            return M

        L_all = []
        for i in range(self.n_agents):
            payout_vec = self.payout_vectors[i]

            # BE CAREFUL WITH SIGNS!
            # OPTIMS REQUIRE LOSSES
            rew_i = torch.matmul(M, payout_vec)
            L_i = -rew_i
            L_all.append(L_i)

        return L_all


def inverse_sigmoid(x):
    return -torch.log((1 / x) - 1)


def init_th(dims, std):
    # th is a list, one entry for each agent, where each entry is the policy for that agent
    th = []
    # Dims [5,5] or something, len of dims is the num of agents
    # And each num represents the dim of the policy for that agent (equal to state space size with binary action/bernoulli dist)
    for i in range(len(dims)):
        if std > 0:
            init = torch.nn.init.normal_(
                torch.empty(dims[i], requires_grad=True), std=std)
        else:
            init = torch.zeros(dims[i], requires_grad=True)
        th.append(init)

    return th


def init_th_uniform(dims):
    th = []
    for i in range(len(dims)):
        init_pols = torch.rand(dims[i], requires_grad=True)
        init_logits = inverse_sigmoid(init_pols)
        th.append(init_logits)
    print("Policies:")
    print(torch.sigmoid(th[0]))
    print(torch.sigmoid(th[1]))
    return th


def init_th_tft(dims, std, logit_shift=3):
    th = []
    for i in range(len(dims)):
        if std > 0:
            init = torch.nn.init.normal_(
                torch.empty(dims[i], requires_grad=True), std=std) - logit_shift
        else:
            init = torch.zeros(dims[i], requires_grad=True) - logit_shift
        init[-1] += 2 * logit_shift
        init[-2] += 2 * logit_shift
        th.append(init)

    return th


class NeuralNet(nn.Module):
    def add_nonlinearity(self, layers):
        if args.nonlinearity == 'lrelu':
            layers.append(torch.nn.LeakyReLU(
                negative_slope=0.01))
        elif args.nonlinearity == 'tanh':
            layers.append(torch.nn.Tanh())
        else:
            raise Exception("No nonlinearity")

    def __init__(self, input_size, hidden_size, extra_hidden_layers,
                 output_size, final_sigmoid=False, final_softmax=False):

        super(NeuralNet, self).__init__()
        layers = []

        layers.append(torch.nn.Linear(input_size, hidden_size))
        self.add_nonlinearity(layers)
        for i in range(extra_hidden_layers):
            layers.append(nn.Linear(hidden_size, hidden_size))
            self.add_nonlinearity(layers)
        layers.append(nn.Linear(hidden_size, output_size))

        if final_sigmoid:
            layers.append(nn.Sigmoid())
        elif final_softmax:
            layers.append(nn.Softmax(dim=-1))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        output = self.net(x)
        return output


class ConvFC(nn.Module):
    def __init__(self, conv_in_channels, conv_out_channels, input_size,
                 hidden_size, output_size, kernel_size=5, final_sigmoid=False):
        super(ConvFC, self).__init__()

        self.conv_out_channels = conv_out_channels

        self.layer1 = nn.Conv2d(conv_in_channels, conv_out_channels,
                                kernel_size=kernel_size)
        self.conv_result_size = (
                    input_size - kernel_size + 1)  # no stride or pad here
        self.fc_size = self.conv_result_size ** 2 * self.conv_out_channels
        self.layer2 = nn.Linear(self.fc_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, output_size)

        self.final_sigmoid = final_sigmoid

    def forward(self, x):

        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        conv_output = torch.tanh(self.layer1(x))
        output = conv_output.reshape(-1, self.fc_size)
        output = torch.tanh(self.layer2(output))
        output = self.layer3(output)

        if self.final_sigmoid:
            output = torch.sigmoid(output)

        return output


def param_init_custom(policy_net, lst):
    counter = 0
    for param in policy_net.parameters():
        param.data = torch.tensor(lst[counter]).requires_grad_()
        counter += 1


def custom_params1():
    l1 = [[-1., 1, -1, 1, -1, 1],
          [-1., 1, -1, 1, -1, 1]]
    l2 = [0., 0]
    l3 = [[-1., 1]]
    l4 = [0.]
    return [l1, l2, l3, l4]


def custom_params2():
    l1 = [[2., -2, 2, -2, 2, -2],
          [2., -2, 2, -2, 2, -2]]
    l2 = [0., 0]
    l3 = [[-2., 2]]
    l4 = [0.]
    return [l1, l2, l3, l4]


def custom_params3():
    l1 = [[10., -10, 10, -10, 10, -10],
          [10., -10, 10, -10, 10, -10]]
    l2 = [0., 0]
    l3 = [[-10., 10]]
    l4 = [0.]
    return [l1, l2, l3, l4]


def custom_params4():
    l1 = [[1.1074687263023, 0.275260757781574, 0.489599039857568,
           0.130220650429065, 0.130220650428934, -0.0416038405697266],
          [0.0142691805853789, 0.933207934425381, -0.0153849853273826,
           0.170635830969251, 0.17063583096918, 0.985901625511033]]
    l2 = [0.286705849, -0.00180708]
    l3 = [[1, 0.5]]
    l4 = [-1.]
    return [l1, l2, l3, l4]


def custom_params5():
    l1 = [[2.76875177202519, 1.70283053656467, 4, 1.36349157832529,
           3.10003656119828, 1],
          [0.515125115736217, 0.51494364232833, 1, 0.289569911385246,
           0.289593098155816, 1]]
    l2 = [1.62568597316727, 0]
    l3 = [[-0.4, 0.6]]
    l4 = [0.]
    return [l1, l2, l3, l4]


def custom_params6():
    l1 = [[1, 2, 4, -3, -21225.4820706105, 1],
          [-0.453054202889886, -0.453054202889869, 1, -1.63518194539645,
           -1.63518176759075, 0.981576517439697]]
    l2 = [-854.61133215301, 1.13967597803765]
    l3 = [[-5.1, 6.9]]
    l4 = [0.]
    return [l1, l2, l3, l4]


def custom_params7():
    l1 = [[-0.734033738503211, 0.381042660301521, 0.445335913409754,
           0.259207802066714, -0.786281879857001, 0.24595939105466],
          [0.20005453974748, 0.200061535711064, -0.249821855722634,
           -0.209748559258265, -0.209750444324398, 0.24013508414252]]
    l2 = [-7.54119786671168, -1.00078349130788]
    l3 = [[-0.36, 0.47]]
    l4 = [0.]
    return [l1, l2, l3, l4]


def init_custom(dims, using_nn=True, nn_hidden_size=16,
                nn_extra_hidden_layers=0):
    th = []
    # NN/func approx
    if using_nn:
        for i in range(len(dims)):

            policy_net = NeuralNet(input_size=dims[i],
                                   hidden_size=nn_hidden_size,
                                   extra_hidden_layers=nn_extra_hidden_layers,
                                   output_size=1)

            if args.custom_param != 'random':
                if args.custom_param == 'mix':
                    if i == 0:
                        lst = custom_params1()
                    elif i == 1:
                        lst = custom_params3()
                    # This could be made dynamic instead of hardcoded like this
                else:
                    # Yeah I know this is ugly
                    if args.custom_param == '1':
                        lst = custom_params1()
                    elif args.custom_param == '2':
                        lst = custom_params2()
                    elif args.custom_param == '3':
                        lst = custom_params3()
                    elif args.custom_param == '4':
                        lst = custom_params4()
                    elif args.custom_param == '5':
                        lst = custom_params5()
                    elif args.custom_param == '6':
                        lst = custom_params6()
                    elif args.custom_param == '7':
                        lst = custom_params7()
                param_init_custom(policy_net, lst)

            th.append(policy_net)

    # Tabular policies
    else:
        for i in range(len(dims)):
            # DONT FORGET THIS +1
            # Right now if you omit the +1 we get a bug where the first state is the prob in the all contrib state
            th.append(torch.nn.init.normal_(
                torch.empty(2 ** n_agents + 1, requires_grad=True),
                std=args.std))

    assert len(th) == len(dims)

    return th


def get_torch_optim_func(optim_type):
    if optim_type.lower() == "sgd":
        return torch.optim.SGD
    elif optim_type.lower() == "adam":
        def get_adam_w_betas(params, lr):
            return torch.optim.Adam(params, lr, betas=(0., 0.99))
        return get_adam_w_betas
    elif optim_type.lower() == "adagrad":
        return torch.optim.Adagrad
    else:
        raise NotImplementedError


def construct_diff_optims(th_or_vals, lrs, f_th_or_vals):
    optims = []
    for i in range(len(th_or_vals)):
        if not isinstance(th_or_vals[i], torch.Tensor):
            optim = get_torch_optim_func(args.optim)(th_or_vals[i].parameters(),
                                                     lr=lrs[i])
            diffoptim = higher.get_diff_optim(optim, th_or_vals[i].parameters(),
                                              f_th_or_vals[i])
            optims.append(diffoptim)
        else:
            # Don't use for now with tabular, not tested
            raise NotImplementedError
            print("Warning: be careful here")
            optim = get_torch_optim_func(args.optim)([th_or_vals[i]], lr=lrs[i])
            diffoptim = higher.optim.DifferentiableSGD(optim, [th_or_vals[i]])
            optims.append(diffoptim)
    return optims


def construct_optims(th_or_vals, lrs):
    optims = []
    for i in range(len(th_or_vals)):
        if not isinstance(th_or_vals[i], torch.Tensor):
            optim = get_torch_optim_func(args.optim)(th_or_vals[i].parameters(),
                                                     lr=lrs[i])
            optims.append(optim)
        else:
            # Don't use for now with tabular, not tested
            raise NotImplementedError
            print("Warning: be careful here")
            optim = get_torch_optim_func(args.optim)([th_or_vals[i]], lr=lrs[i])
            optims.append(optim)
    return optims


def get_gradient(function, param):
    grad = torch.autograd.grad(function, param, create_graph=True)[0]
    return grad


def get_jacobian(terms, param):
    jac = []
    for term in terms:
        grad = \
        torch.autograd.grad(term, param, retain_graph=True, create_graph=False)[
            0]
        jac.append(grad.flatten())
    jac = torch.vstack(jac)
    return jac


def get_th_copy(th, dims_to_use=None):
    if dims_to_use is None:
        dims_to_use = dims
    new_th = init_custom(dims_to_use, True,
                         args.nn_hidden_size, args.nn_extra_hidden_layers)

    for i in range(len(th)):
        # print(f"---{i}---")
        # print(th[i])
        # print(isinstance(th[i], NeuralNet))
        if isinstance(th[i], NeuralNet):
            copyNN(new_th[i], th[i])
        else:
            new_th[i] = copy.deepcopy(th[i])

    return new_th

    # if isinstance(th[0], NeuralNet):
    #     new_th = init_custom(dims, args.using_nn,
    #                          args.nn_hidden_size, args.nn_extra_hidden_layers)
    #     for i in range(len(th)):
    #         copyNN(new_th[i], th[i])
    #     return new_th
    # else:
    #     return copy.deepcopy(th)


def build_policy_dist(coop_probs):
    # This version just for ipdn/exact.
    defect_probs = 1 - coop_probs

    policy_dist = torch.vstack((coop_probs, defect_probs)).t()

    # we need to do this because kl_div needs the full distribution
    # and the way we have parameterized policy here is just a coop prob
    # if you used categorical/multinomial you wouldn't have to go through this
    # The way torch kldiv works is that the first dimension is the batch, the last dimension is the probabilities.
    # The reshape just makes so that batchmean occurs over the first axis

    policy_dist = policy_dist.reshape(1, -1, 2)
    return policy_dist


def build_policy_and_target_policy_dists(policy_to_build, target_pol_to_build,
                                         i, policies_are_logits=True, ill_cond=False):
    # Note the policy and targets are individual agent ones
    # Only used in tabular case so far

    if ill_cond:
        print("BE CAREFUL that the policies passed in are not already adjusted for the ill-cond")

    if policies_are_logits:
        # if args.ill_condition:
        if ill_cond:
            policy_dist = build_policy_dist(
                torch.sigmoid(ill_cond_matrices[i] @ policy_to_build))

            target_policy_dist = build_policy_dist(
                torch.sigmoid(
                    ill_cond_matrices[i] @ target_pol_to_build.detach()))
        else:
            policy_dist = build_policy_dist(torch.sigmoid(policy_to_build))
            target_policy_dist = build_policy_dist(
                torch.sigmoid(target_pol_to_build.detach()))
    else:
        policy_dist = build_policy_dist(policy_to_build)
        target_policy_dist = build_policy_dist(target_pol_to_build.detach())
    return policy_dist, target_policy_dist

def get_kl_div_from_policies(policy, target_policy, i, policies_are_logits=False):
    policy_dist, target_policy_dist = build_policy_and_target_policy_dists(
                                policy, target_policy, i, policies_are_logits=policies_are_logits,
                                ill_cond=args.ill_condition)
    kl_div_reduction = 'batchmean'
    kl_div = torch.nn.functional.kl_div(
        input=torch.log(policy_dist),
        target=target_policy_dist,
        reduction=kl_div_reduction,
        log_target=False)
    return kl_div

def get_discounted_state_visitation_weighted_kl(kl_div_no_reduce, p_mat):
    weighted_kl_div = (
                kl_div_no_reduce.sum(dim=-1) * p_mat.reshape(1, -1)).mean()
    weighted_kl_div = weighted_kl_div / p_mat.sum()
    return weighted_kl_div





def prox_f(th_to_build_on, kl_div_target_th, game, i, j, prox_f_step_sizes,
           iters=0, max_iters=10000, ill_cond=False):
    # For each other player, do the prox operator
    # (this function just does on a single player, it should be used within the loop iterating over all players)
    # We will do this by gradient descent on the proximal objective
    # Until we reach a fixed point, which tells use we have reached
    # the minimum of the prox objective, which is our prox operator result
    # i is the self, j is the other agent index. Just used for the ill cond on OM only...
    fixed_point_reached = False

    new_pol = game.get_policy_for_all_states(th_to_build_on, j, ill_cond=ill_cond)
    curr_pol = new_pol.detach().clone()

    # print(new_pol)
    # print(torch.sigmoid(th_to_build_on[j]))

    while not fixed_point_reached:

        inner_losses = game.get_exact_loss(th_to_build_on, ill_cond=ill_cond, self_no_cond=(not args.ill_condition), self_index=i)
        # inner_losses = game.get_exact_loss(th_to_build_on, ill_cond=ill_cond)

        policy = game.get_policy_for_all_states(th_to_build_on, j, ill_cond=ill_cond)
        target_policy = game.get_policy_for_all_states(kl_div_target_th, j, ill_cond=ill_cond)

        # print("DEBUGG")
        # print(policy)
        # print(target_policy)

        policy_dist, target_policy_dist = build_policy_and_target_policy_dists(
            policy, target_policy, j, policies_are_logits=False, ill_cond=False)

        # print("DEBUGG")
        # print(policy_dist)
        # print(target_policy_dist)

        kl_div_reduction = 'batchmean'

        if args.visitation_weighted_kl:
            kl_div_reduction = 'none'

            p_mat = game.get_exact_loss(th_to_build_on, return_p_mat_only=True, ill_cond=ill_cond, self_no_cond=(not args.ill_condition), self_index=i)
            if args.init_state_representation == 2:
                p_mat = torch.cat((p_mat, torch.ones(1)))
            elif args.init_state_representation == 1:
                p_mat[-1] += 1
                p_mat = torch.cat((p_mat, torch.zeros(1)))
            else:
                raise NotImplementedError


        kl_div = torch.nn.functional.kl_div(input=torch.log(policy_dist),
                                            target=target_policy_dist,
                                            reduction=kl_div_reduction,
                                            log_target=False)

        if args.visitation_weighted_kl:
            kl_div = get_discounted_state_visitation_weighted_kl(kl_div, p_mat)

        loss_j = inner_losses[j] + args.inner_beta * kl_div
        # No eta here because we are going to solve it in the loop with many iterations anyway

        # Non-diff to make it nl loss on outer loop, and we will use the other_terms from ift
        # (basically no need for grad through the optim process, we only need the fixed point, and will calculate grad there for IFT)
        combined_grad_squared = 0

        with torch.no_grad():
            if isinstance(th_to_build_on[j], NeuralNet):
                for param in th_to_build_on[j].parameters():
                    param_grad = get_gradient(loss_j, param)
                    combined_grad_squared += (param_grad ** 2).sum()
                    param.data -= prox_f_step_sizes[j] * param_grad

            else:
                pol_grad = get_gradient(loss_j, th_to_build_on[j])
                combined_grad_squared += (pol_grad ** 2).sum()
                th_to_build_on[j] -= prox_f_step_sizes[j] * pol_grad

        combined_grad_l2 = combined_grad_squared ** (1./2.)

        prev_pol = curr_pol.detach().clone()
        new_pol = game.get_policy_for_all_states(th_to_build_on, j, ill_cond=ill_cond)
        curr_pol = new_pol.detach().clone()

        # policy_dist, target_policy_dist = build_policy_and_target_policy_dists(
        #     curr_pol, prev_pol, j, policies_are_logits=False, ill_cond=ill_cond)
        #
        # curr_prev_pol_div = torch.nn.functional.kl_div(
        #     input=torch.log(policy_dist),
        #     target=target_policy_dist,
        #     reduction=kl_div_reduction,
        #     log_target=False)
        #
        # curr_prev_pol_div_rev = torch.nn.functional.kl_div(
        #     input=torch.log(target_policy_dist),
        #     target=policy_dist,
        #     reduction=kl_div_reduction,
        #     log_target=False)




        # if args.visitation_weighted_kl:
        #     # This stuff was originally supposed to help with numerical precision issues but I actually think it doesn't make that much of a difference
        #     curr_prev_pol_div = get_discounted_state_visitation_weighted_kl(
        #         curr_prev_pol_div, p_mat)
        #     curr_prev_pol_div_rev = get_discounted_state_visitation_weighted_kl(
        #         curr_prev_pol_div_rev, p_mat)

        # print(f"--INNER STEP {iters}--")
        # print("Prev pol:")
        # print(prev_pol)
        # print("Curr pol:")
        # print(curr_pol)
        # print(combined_grad_l2)

        iters += 1

        l2_threshold = 1e-5
        # if (curr_prev_pol_div < threshold and curr_prev_pol_div_rev < threshold) or iters > max_iters:
        if combined_grad_l2 < l2_threshold or iters > max_iters:
            if args.print_prox_loops_info:
                print("Inner prox iters used: {}".format(iters))
            if iters >= max_iters:
                print("Reached max prox iters")
                print(combined_grad_l2)
            fixed_point_reached = True

            print(f"--INNER STEP {iters}--")
            print("Prev pol:")
            print(prev_pol)
            print("Curr pol:")
            print(curr_pol)
            print(combined_grad_l2)

    if isinstance(th_to_build_on[j], NeuralNet):
        if args.opp_model and args.om_using_nn:
            dims_to_use = om_dims
        else:
            dims_to_use = dims

        return get_th_copy(th_to_build_on, dims_to_use=dims_to_use)[j]
    else:
        return th_to_build_on[j].detach().clone().requires_grad_()


# Everything that passes game as a parameter can instead be moved into the class itself
# and made as a method of that class...
def get_ift_terms(inner_lookahead_th, kl_div_target_th, game, i, j, ill_cond=False):
    losses_for_ift = game.get_exact_loss(
        inner_lookahead_th, ill_cond=ill_cond, self_no_cond=(not args.ill_condition), self_index=i)  # Note that new_th has only agent j updated

    if isinstance(inner_lookahead_th[j], NeuralNet):

        grad2_V1 = []
        for param in inner_lookahead_th[j].parameters():
            g = get_gradient(losses_for_ift[i], param)
            grad2_V1.append(g.flatten())

        grad2_V1 = torch.cat(grad2_V1)

    else:
        grad2_V1 = get_gradient(losses_for_ift[i], inner_lookahead_th[j])

    # We use inner_lookahead_th instead of new_th because inner_lookahead has only th[j] updated
    policy = game.get_policy_for_all_states(inner_lookahead_th, j, ill_cond=ill_cond)
    target_policy = game.get_policy_for_all_states(kl_div_target_th, j, ill_cond=ill_cond)

    policy_dist, target_policy_dist = build_policy_and_target_policy_dists(
        policy, target_policy, j, policies_are_logits=False, ill_cond=False)

    kl_div = torch.nn.functional.kl_div(
        input=torch.log(policy_dist),
        target=target_policy_dist,
        reduction='batchmean',
        log_target=False)

    loss_j = losses_for_ift[j] + args.inner_beta * kl_div
    # LR here will have an effect on the outer gradient update
    if isinstance(inner_lookahead_th[j], NeuralNet):
        f_at_fixed_point = []
        for param in inner_lookahead_th[j].parameters():
            f_at_fixed_point.append(
                param - lr_policies_inner[j] * get_gradient(loss_j, param))

    else:
        f_at_fixed_point = inner_lookahead_th[j] - lr_policies_inner[
            j] * get_gradient(
            loss_j, inner_lookahead_th[j])

    print_info = args.print_prox_loops_info
    if print_info:
        game.print_policies_for_all_states(inner_lookahead_th, ill_cond=ill_cond)

    f_at_fixed_point = torch.cat(list(map(torch.flatten, f_at_fixed_point)))

    if isinstance(inner_lookahead_th[i], NeuralNet):
        # new_f_at_fixed_point = torch.cat(list(map(torch.flatten, f_at_fixed_point)))
        grad0_f = []
        for param in inner_lookahead_th[i].parameters():
            jac = get_jacobian(f_at_fixed_point, param)
            grad0_f.append(jac)
        grad0_f = torch.hstack(grad0_f)
    else:
        grad0_f = get_jacobian(f_at_fixed_point, inner_lookahead_th[i])

    if isinstance(inner_lookahead_th[j], NeuralNet):
        # new_f_at_fixed_point = torch.cat(list(map(torch.flatten, f_at_fixed_point)))
        grad1_f = []
        for param in inner_lookahead_th[j].parameters():
            jac = get_jacobian(f_at_fixed_point, param)
            grad1_f.append(jac)
        grad1_f = torch.hstack(grad1_f)
    else:
        grad1_f = get_jacobian(f_at_fixed_point, inner_lookahead_th[j])

    # if isinstance(inner_lookahead_th[j], NeuralNet):
    #     f_at_fixed_point = torch.cat(list(map(torch.flatten, f_at_fixed_point)))
    #     grad0_f = []
    #     for param in inner_lookahead_th[i].parameters():
    #         jac = get_jacobian(f_at_fixed_point, param)
    #         grad0_f.append(jac)
    #     grad0_f = torch.hstack(grad0_f)
    #
    #     grad1_f = []
    #     for param in inner_lookahead_th[j].parameters():
    #         jac = get_jacobian(f_at_fixed_point, param)
    #         grad1_f.append(jac)
    #     grad1_f = torch.hstack(grad1_f)
    #
    # else:
    #     grad0_f = get_jacobian(f_at_fixed_point, inner_lookahead_th[i])
    #     grad1_f = get_jacobian(f_at_fixed_point, inner_lookahead_th[j])


    # th[j], th[j] shape for grad1_f
    # th[j], th[i] shape for grad0_f

    # This inverse can fail sometimes.
    # This is the main issue with the IFT based POLA version and why it didn't make the final paper (well, that, plus the long running time).
    # The IFT requires that the Hessian be nonsingular, otherwise the inverse won't exist.
    # If the policy network is overparameterized, the Hessian will necessarily be singular.
    mat_to_inv = torch.eye(grad1_f.shape[0]) - grad1_f

    grad_th1_th2prime = torch.inverse(mat_to_inv) @ grad0_f

    return grad2_V1, grad_th1_th2prime


def inner_exact_loop_step(starting_th, kl_div_target_th, game, i, n,
                          prox_f_step_sizes, opp_models = None, optims_th_primes = None):

    other_terms = []

    # if args.opp_model:
    #     new_th = get_th_copy(starting_th, dims_to_use=om_dims)
    # else:
    #     new_th = get_th_copy(starting_th)

    # i is the agent doing the rolling out of the other agents
    for j in range(n):
        if j != i:
            # Inner lookahead th has only agent j's th being updated
            if args.opp_model:
                inner_lookahead_th = get_th_copy(starting_th, dims_to_use=om_dims)
                new_th = get_th_copy(starting_th, dims_to_use=om_dims)

            else:
                inner_lookahead_th = get_th_copy(starting_th)
                new_th = get_th_copy(starting_th)

            # inner_lookahead_th = get_th_copy(starting_th)

            # Inner loop essentially
            # Each player on the copied th does a naive update (doesn't have to be differentiable here because of fixed point/IFT)

            if args.opp_model:
                ill_cond = args.om_precond
            else:
                ill_cond = args.ill_condition

            # For each other player, do the prox operator

            # new_th[j] = inner_lookahead_th[j].detach().clone().requires_grad_()
            # You could do this without the detach, and using x = x - grad instead of -= above, and no torch.no_grad
            # And then differentiate through the entire unrolled process
            # But we want instead fixed point / IFT for efficiency and not having to differentiate through entire unroll process.
            # For nice IFT primer, check out https://implicit-layers-tutorial.org/implicit_functions/


            inner_lookahead_th[j] = prox_f(inner_lookahead_th,
                                           kl_div_target_th,
                                           game, i, j, prox_f_step_sizes,
                                           max_iters=args.prox_inner_max_iters,
                                           ill_cond=ill_cond)

            grad2_V1, grad_th1_th2prime = get_ift_terms(inner_lookahead_th,
                                                        kl_div_target_th,
                                                        game, i, j, ill_cond=ill_cond)

            other_terms.append(grad2_V1 @ grad_th1_th2prime)


            new_th[j] = inner_lookahead_th[j]

    return new_th, other_terms

def inner_loop_step(game, th_with_only_agent_i_updated, starting_th, i, lr_policies_inner, lr_policies_outer, optims_th_primes=None):
    # Do differentiable updates and produce a new_th after all agents updated
    new_th = get_th_copy(th_with_only_agent_i_updated)

    for j in range(len(th_with_only_agent_i_updated)):
        if j != i:

            inner_lookahead_th = get_th_copy(th_with_only_agent_i_updated)

            for inner_step in range(args.prox_inner_max_iters):
                if args.opp_model:
                    if args.ill_condition:
                        inner_losses = game.get_exact_loss(inner_lookahead_th,
                                                           ill_cond=args.om_precond)
                    else:
                        inner_losses = game.get_exact_loss(inner_lookahead_th,
                                                           ill_cond=args.om_precond,
                                                           self_no_cond=True,
                                                           self_index=i)
                else:
                    inner_losses = game.get_exact_loss(inner_lookahead_th,
                                                       ill_cond=args.ill_condition)

                # print("TESTING ONLY")
                # game.print_policies_for_all_states(new_th, ill_cond=args.om_precond)

                if args.opp_model:
                    use_ill_cond = args.om_precond
                else:
                    use_ill_cond = args.ill_condition


                policy = game.get_policy_for_all_states(
                    inner_lookahead_th, j, ill_cond=use_ill_cond)
                target_policy = game.get_policy_for_all_states(
                    starting_th, j,
                    ill_cond=use_ill_cond)
                kl_div = get_kl_div_from_policies(policy, target_policy, j)
                inner_losses[j] += kl_div * args.inner_beta

                if isinstance(inner_lookahead_th[j], torch.Tensor):
                    inner_lookahead_th[j] = inner_lookahead_th[j] - lr_policies_inner[
                        j] * get_gradient(inner_losses[j],
                                          inner_lookahead_th[j])
                else:
                    assert optims_th_primes is not None
                    optim_update(optims_th_primes[j],
                                 inner_losses[j],
                                 inner_lookahead_th[j].parameters())

                # print("TESTING ONLY")
                # game.print_policies_for_all_states(new_th,
                #                                    ill_cond=args.om_precond)
            new_th[j] = inner_lookahead_th[j]

    return new_th


# print("AFTER INNER UPDATES")
# game.print_policies_for_all_states(new_th, ill_cond=args.om_precond)


def outer_exact_loop_step(print_info, i, new_th, static_th_copy, game, curr_pol,
                          other_terms, curr_iter, optims_th_primes=None):
    if print_info:
        game.print_policies_for_all_states(new_th, ill_cond=args.ill_condition)

    outer_loss = game.get_exact_loss(new_th, ill_cond=args.ill_condition, self_no_cond=(not args.ill_condition), self_index=i)

    policy = game.get_policy_for_all_states(new_th, i, ill_cond=args.ill_condition)
    target_policy = game.get_policy_for_all_states(static_th_copy, i, ill_cond=args.ill_condition)

    policy_dist, target_policy_dist = build_policy_and_target_policy_dists(
        policy, target_policy, i, policies_are_logits=False, ill_cond=False)

    kl_div = torch.nn.functional.kl_div(
        input=torch.log(policy_dist),
        target=target_policy_dist,
        reduction='batchmean',
        log_target=False)

    loss_i = outer_loss[i] + args.outer_beta * kl_div

    with torch.no_grad():
        if isinstance(new_th[i], torch.Tensor):
            if other_terms is not None:
                new_th[i] -= lr_policies_outer[i] * (
                        get_gradient(loss_i, new_th[i]) + sum(other_terms))
            else:
                new_th[i] -= lr_policies_outer[i] * (
                    get_gradient(loss_i, new_th[i]))
        else:
            assert optims_th_primes is not None

            optim_update(optims_th_primes[i], loss_i, new_th[i].parameters())

            if other_terms is not None:
                counter = 0
                sum_terms = sum(other_terms)
                for param in new_th[i].parameters():
                    param_len = param.flatten().shape[0]
                    term_to_add = sum_terms[counter: counter + param_len]
                    param.data -= lr_policies_outer[i] * term_to_add.reshape(
                        param.shape)
                    counter += param_len

    prev_pol = curr_pol.detach().clone()
    new_pol = game.get_policy_for_all_states(new_th, i, ill_cond=args.ill_condition)
    curr_pol = new_pol.detach().clone()

    if print_info:
        print(kl_div)
        print("Prev pol")
        print(prev_pol)
        print("Curr pol")
        print(curr_pol)

        print("Iter:")
        print(curr_iter)

    policy_dist, target_policy_dist = build_policy_and_target_policy_dists(
        curr_pol, prev_pol, i, policies_are_logits=False, ill_cond=False)

    curr_prev_pol_div = torch.nn.functional.kl_div(
        input=torch.log(policy_dist),
        target=target_policy_dist,
        reduction='batchmean',
        log_target=False)

    curr_prev_pol_div_rev = torch.nn.functional.kl_div(
        input=torch.log(target_policy_dist),
        target=policy_dist,
        reduction='batchmean',
        log_target=False)

    fixed_point_reached = False
    # This checking 2 ways was originally supposed to help with numerical precision issues but I actually think it doesn't make that much of a difference
    if (curr_prev_pol_div < args.prox_threshold and curr_prev_pol_div_rev < args.prox_threshold) or curr_iter > args.prox_outer_max_iters:
        print("Outer prox iters used: {}".format(curr_iter))
        if curr_iter >= args.prox_outer_max_iters:
            print("Reached max prox iters")
        fixed_point_reached = True

    return new_th, curr_pol, fixed_point_reached


def print_exact_policy(th, i, ill_cond=False):
    # Used for exact gradient setting
    print(
        "---Agent {} Rollout---".format(i + 1))
    for j in range(len(th)):
        print("Agent {} Policy".format(j + 1), flush=True)
        print(torch.sigmoid(th[j]))

        # if args.ill_condition:
        if ill_cond:
            print("Agent {} Transformed Policy".format(j + 1))
            print(torch.sigmoid(ill_cond_matrices[j] @ th[j]))



def list_dot(l1, l2):
    # Essentially does a dot product, but instead of tensors or arrays, works with a list of elements
    # assumes same length of lists
    assert len(l1) == len(l2)
    sum = 0
    for i in range(len(l1)):
        sum += torch.sum(l1[i] * l2[i])
    return sum


def update_th_taylor_approx_exact_value(th, game):
    # This is the original LOLA formulation with their unnecessary Taylor approximation
    # even though you could just directly calculate the update without the approximation
    assert not args.actual_update
    n = len(th)

    losses = game.get_exact_loss(th, ill_cond=args.ill_condition)

    # Compute gradients
    # This is a 2d array of all the pairwise gradient computations between all agents
    # This is useful for LOLA and the other opponent modeling stuff
    # So it is the gradient of the loss of agent j with respect to the parameters of agent i
    # When j=i this is just regular gradient update

    if args.using_nn:
        total_params = 0
        for param in th[0].parameters():
            total_params += 1

        grad_L = [[[0] * total_params] * n] * n

        for i in range(n):
            for j in range(n):
                k = 0
                for param in th[i].parameters():
                    grad = get_gradient(losses[j], param)
                    grad_L[i][j][k] = grad
                    k += 1

    else:
        grad_L = [[get_gradient(losses[j], th[i]) for j in range(n)] for i in
                  range(n)]

    # calculate grad_L as the gradient of loss for player j with respect to parameters for player i
    # Therefore grad_L[i][i] is simply the naive learning loss for player i

    # Be careful with mixed algorithms here; I have not tested it much

    print_info = args.print_prox_loops_info

    if args.inner_exact_prox:
        # Do this later. Not used in the paper anyway
        raise NotImplementedError

    else:
        # Here we continue with the exact gradient calculations
        # Look at pg 12, Stable Opponent Shaping paper
        # This is the first line of the second set of equations
        # sum over all j != i of grad_j L_i * grad_j L_j
        # Then that's your lola term once you differentiate it with respect to theta_i
        # Then add the naive learning term
        if args.using_nn:
            terms = [sum([list_dot(grad_L[j][i], grad_L[j][j])
                          for j in range(n) if j != i]) for i in range(n)]

            lola_terms = [[0] * total_params] * n
            for i in range(n):
                k = 0
                for param in th[i].parameters():
                    lola_terms[i][k] = lr_policies_inner[i] * -get_gradient(
                        terms[i], param)
                    k += 1
            nl_terms = [grad_L[i][i] for i in range(n)]

        else:

            terms = [sum([torch.dot(grad_L[j][i], grad_L[j][j])
                          for j in range(n) if j != i]) for i in range(n)]

            lola_terms = [
                lr_policies_inner[i] * -get_gradient(terms[i], th[i])
                for i in range(n)]

            nl_terms = [grad_L[i][i] for i in range(n)]

            grads = [nl_terms[i] + lola_terms[i] for i in range(n)]

    # Update theta
    with torch.no_grad():
        for i in range(n):
            if not isinstance(th[i], torch.Tensor):
                k = 0
                for param in th[i].parameters():
                    param -= lr_policies_outer[i] * (
                                nl_terms[i][k] + lola_terms[i][k])
                    k += 1
            else:
                th[i] -= lr_policies_outer[i] * grads[i]

    return th



def get_new_th_and_optims(th, n_agents, i, lr_policies_outer, lr_policies_inner, static_th_copy, om_static_copy=None):
    optims_th_primes = None
    if args.opp_model:
        assert om_static_copy is not None
        if args.om_using_nn:
            new_th, optims_th_primes = \
                construct_f_th_and_diffoptim(n_agents, i,
                                             om_static_copy[i],
                                             lr_policies_outer,
                                             lr_policies_inner
                                             )
            new_th[i] = th[i]  # th here is being updated

            if args.using_nn:
                raise NotImplementedError  # TODO Perhaps just need th_with_only_agent_i_updated again here

            # new_th[i] = static_th_copy[i] # replace an opp model of self with the actual self model

        else:
            new_th = get_th_copy(om_static_copy[i])
            new_th[i] = th[i]  # replace an opp model of self with the actual self model
            # new_th[i] = static_th_copy[i] # replace an opp model of self with the actual self model

    else:
        if args.using_nn:
            # if args.outer_steps > 1:
            #     raise NotImplementedError
            th_with_only_agent_i_updated = copy.deepcopy(
                static_th_copy)
            th_with_only_agent_i_updated[i] = th[i]

            new_th, optims_th_primes = \
                construct_f_th_and_diffoptim(n_agents, i,
                                             th_with_only_agent_i_updated,
                                             lr_policies_outer,
                                             lr_policies_inner
                                             )


        else:
            new_th = get_th_copy(static_th_copy)
            new_th[i] = th[i]

    return new_th, optims_th_primes



def update_th_exact_value_outer_exact_prox(th, game, opp_models=None):
    # opp_models is a list of lists, where each sublist is basically a th that
    # the agent i thinks all the other agents are using

    if args.opp_model:
        assert opp_models is not None

    assert args.actual_update
    # Do DiCE style rollouts except we can calculate exact Ls like follows
    if opp_models is not None:
        om_static_copy = get_th_copy(opp_models)
        raise NotImplementedError

    static_th_copy = get_th_copy(th)
    n = len(th)

    for i in range(n):

        fixed_point_reached = False
        outer_iters = 0

        curr_pol = game.get_policy_for_all_states(th, i,
                                                  ill_cond=args.ill_condition).detach()  # just an initialization, will be updated in the outer step loop, ill_cond=args.ill_condition)

        optims_th_primes = None
        if args.inner_exact_prox:
            if args.using_nn:
                # Do only once if we have inner_exact_prox
                th_with_only_agent_i_updated = copy.deepcopy(
                    static_th_copy)
                th_with_only_agent_i_updated[i] = th[i]

                new_th, optims_th_primes = \
                    construct_f_th_and_diffoptim(n_agents, i,
                                                 th_with_only_agent_i_updated,
                                                 lr_policies_outer,
                                                 lr_policies_inner
                                                 )
            else:
                new_th = get_th_copy(static_th_copy)
                new_th[i] = th[i]

        while (not fixed_point_reached) and (outer_iters < args.prox_outer_max_iters):

            # --- INNER LOOP ---
            other_terms = None
            if args.inner_exact_prox:
                temp_new_th, other_terms = inner_exact_loop_step(new_th,
                                                                 static_th_copy,
                                                                 game, i, n,
                                                                 prox_f_step_sizes=lr_policies_inner,
                                                                 optims_th_primes=optims_th_primes)

                # numerical_issue = False
                # try:
                #     new_pol = game.get_policy_for_all_states(temp_new_th, i,
                #                                              ill_cond=args.ill_condition)
                #     prev_pol = game.get_policy_for_all_states(new_th, i,
                #                                               ill_cond=args.ill_condition)
                #
                #     policy_dist, target_policy_dist = build_policy_and_target_policy_dists(
                #         new_pol, prev_pol, i, policies_are_logits=False)
                #
                #     kl_div_error_check = torch.nn.functional.kl_div(
                #         input=torch.log(policy_dist),
                #         target=target_policy_dist,
                #         reduction='batchmean',
                #         log_target=False)
                #
                #     print(kl_div_error_check)
                #
                # except:
                #     numerical_issue = True
                #
                # if numerical_issue or kl_div_error_check > 20:
                #     print("Numerical Error occured")
                #     print(kl_div_error_check)
                #     continue
                # else:
                #     new_th = temp_new_th

                new_th = temp_new_th

            else:
                if args.opp_model:
                    raise NotImplementedError
                    # remember th has only agent i being updated
                    new_th, optims_th_primes = \
                        get_new_th_and_optims(th, n_agents, i,
                                              lr_policies_outer,
                                              lr_policies_inner, static_th_copy,
                                              om_static_copy)

                else:
                    new_th, optims_th_primes = \
                        get_new_th_and_optims(th, n_agents, i,
                                              lr_policies_outer,
                                              lr_policies_inner, static_th_copy)
                    # optims_th_primes will be none if not args.using_nn


                # TODO see if this needs something like self_no_cond=(not args.ill_condition), self_index=i
                inner_losses = game.get_exact_loss(new_th,
                                                   ill_cond=args.ill_condition)

                for j in range(n):
                    # Inner loop step essentially
                    # Each player on the copied th does a naive update (must be differentiable!)
                    if j != i:
                        if isinstance(new_th[j], torch.Tensor):
                            new_th[j] = new_th[j] - lr_policies_inner[
                                j] * get_gradient(inner_losses[j],
                                                  new_th[j])
                        else:
                            optim_update(optims_th_primes[j],
                                         inner_losses[j],
                                         new_th[j].parameters())

            outer_iters += 1
            if args.using_nn:
                new_th, curr_pol, fixed_point_reached = outer_exact_loop_step(
                    args.print_prox_loops_info, i, new_th, static_th_copy,
                    game,
                    curr_pol, other_terms, outer_iters,
                    optims_th_primes)
            else:
                new_th, curr_pol, fixed_point_reached = outer_exact_loop_step(
                    args.print_prox_loops_info, i, new_th, static_th_copy,
                    game, curr_pol, other_terms, outer_iters, None)

            print(f"--OUTER STEP {outer_iters}--")
            print(curr_pol)

            if isinstance(new_th[i], torch.Tensor):
                th[i] = new_th[i]
            else:
                copyNN(th[i], new_th[i])

    return th


def update_th_exact_value_outer_steps(th, game, opp_models=None):
    # opp_models is a list of lists, where each sublist is basically a th that
    # the agent i thinks all the other agents are using

    if args.opp_model:
        assert opp_models is not None

    assert args.actual_update
    # Do DiCE style rollouts except we can calculate exact Ls like follows
    if opp_models is not None:
        om_static_copy = get_th_copy(opp_models)

    static_th_copy = get_th_copy(th)
    n = len(th)

    for i in range(n):

        for outer_step in range(args.outer_steps):

            # TODO rewrite this whole loop. Just make it modular and everything flows nicely
            # Opp model should always be there. If no OM, just have the opp model be a copy of the other agent true policy
            # Then we don't have all these random conditions
            # Just clean up this whole thing. Reproduce the tabular and OM experimental results I had before first
            # Then move on to NN.

            if args.opp_model:
                new_th, optims_th_primes = \
                    get_new_th_and_optims(th, n_agents, i, lr_policies_outer,
                                          lr_policies_inner, static_th_copy, om_static_copy)

            else:
                new_th, optims_th_primes = \
                    get_new_th_and_optims(th, n_agents, i,
                                          lr_policies_outer,
                                          lr_policies_inner, static_th_copy)
                # optims_th_primes will be none if not args.using_nn


            # --- INNER LOOP ---
            # Then each player calcs the losses
            other_terms = None
            if args.inner_exact_prox:
                if args.opp_model:
                    target_th = om_static_copy[i]
                else:
                    target_th = static_th_copy
                temp_new_th, other_terms = inner_exact_loop_step(new_th,
                                                                 target_th,
                                                                 game, i, n,
                                                                 prox_f_step_sizes=lr_policies_inner)
                new_th = temp_new_th

                if args.using_nn:
                    new_th, optims_th_primes = \
                        construct_f_th_and_diffoptim(n_agents, i,
                                                     new_th,
                                                     lr_policies_outer,
                                                     lr_policies_inner
                                                     )

            else:
                if args.opp_model:
                    dims_to_use = om_dims
                else:
                    dims_to_use = dims

                starting_th = get_th_copy(new_th, dims_to_use=dims_to_use)
                for inner_step in range(args.inner_steps):
                    if args.opp_model:
                        if args.ill_condition:
                            inner_losses = game.get_exact_loss(new_th,
                                                               ill_cond=args.om_precond)
                        else:
                            inner_losses = game.get_exact_loss(new_th,
                                                               ill_cond=args.om_precond,
                                                               self_no_cond=True,
                                                               self_index=i)
                    else:
                        inner_losses = game.get_exact_loss(new_th,
                                                           ill_cond=args.ill_condition)

                    # print("TESTING ONLY")
                    # game.print_policies_for_all_states(new_th, ill_cond=args.om_precond)

                    if args.opp_model:
                        use_ill_cond = args.om_precond
                    else:
                        use_ill_cond = args.ill_condition

                    for ii in range(len(th)):
                        policy = game.get_policy_for_all_states(
                            new_th, ii, ill_cond=use_ill_cond)
                        target_policy = game.get_policy_for_all_states(starting_th,
                                                                       ii,
                                                                       ill_cond=use_ill_cond)
                        kl_div = get_kl_div_from_policies(policy, target_policy, ii)
                        inner_losses[ii] += kl_div * args.inner_beta


                    for j in range(n):
                        # Inner loop essentially
                        # Each player on the copied th does a naive update (must be differentiable!)
                        if j != i:
                            if isinstance(new_th[j], torch.Tensor):
                                new_th[j] = new_th[j] - lr_policies_inner[
                                    j] * get_gradient(inner_losses[j],
                                                      new_th[j])
                            else:
                                optim_update(optims_th_primes[j],
                                             inner_losses[j],
                                             new_th[j].parameters())

                    # print("TESTING ONLY")
                    # game.print_policies_for_all_states(new_th,
                    #                                    ill_cond=args.om_precond)
                # print("AFTER INNER UPDATES")
                # game.print_policies_for_all_states(new_th, ill_cond=args.om_precond)

            if args.print_inner_rollouts:
                print_exact_policy(new_th, i, ill_cond=args.ill_condition)

            # Then each player recalcs losses using mixed th where everyone else's is the new th but own th is the old (copied) one (do this in a for loop)

            # print(args.om_precond)
            # print(not args.ill_condition)
            outer_losses = game.get_exact_loss(new_th, ill_cond=args.om_precond,
                                               self_no_cond=(
                                                   not args.ill_condition),
                                               self_index=i)

            if other_terms is not None:
                if args.taylor_with_actual_update:
                    raise NotImplementedError

                if isinstance(new_th[i], torch.Tensor):
                    with torch.no_grad():
                        new_th[i] -= lr_policies_outer[i] * (get_gradient(
                            outer_losses[i], new_th[i]) + sum(other_terms))
                    # Finally we rewrite the th by copying from the created copies
                    th[i] = new_th[i]
                else:
                    print(other_terms)
                    print(outer_losses)
                    game.print_policies_for_all_states(th)

                    optim_update(optims_th_primes[i], outer_losses[i],
                                 new_th[i].parameters())

                    game.print_policies_for_all_states(th)
                    1 / 0

                    counter = 0
                    sum_terms = sum(other_terms)
                    for param in new_th[i].parameters():
                        param_len = param.flatten().shape[0]
                        term_to_add = sum_terms[
                                      counter: counter + param_len]
                        param.data -= lr_policies_outer[
                                          i] * term_to_add.reshape(
                            param.shape)
                        counter += param_len


            else:

                loss_to_update = outer_losses[i]

                policy = game.get_policy_for_all_states(
                    new_th, i, ill_cond=args.ill_condition)
                target_policy = game.get_policy_for_all_states(
                    static_th_copy, i, ill_cond=args.ill_condition)
                kl_div_outer = get_kl_div_from_policies(policy, target_policy,
                                                        i)
                outer_losses[i] += kl_div_outer * args.outer_beta


                # Finally each player updates their own (copied) th
                if isinstance(new_th[i], torch.Tensor):
                    with torch.no_grad():
                        new_th[i] -= lr_policies_outer[i] * get_gradient(
                            loss_to_update, new_th[i])
                    # Finally we rewrite the th by copying from the created copies
                    th[i] = new_th[i]
                else:
                    optim_update(optims_th_primes[i], loss_to_update,
                                 new_th[i].parameters())

                    copyNN(th[i], new_th[i])

            policy = game.get_policy_for_all_states(
                th, i, ill_cond=args.ill_condition)
            print(f"Outer step: {outer_step}")
            print(policy)

    return th

def update_th_exact_value(th, game, opp_models=None):
    # new_th = update_th_exact_value_outer_exact_prox(th, game, opp_models)

    if args.outer_exact_prox:
        new_th = update_th_exact_value_outer_exact_prox(th, game, opp_models)
    else:
        new_th = update_th_exact_value_outer_steps(th, game, opp_models)
    return new_th


def construct_f_th_and_diffoptim(n_agents, i, starting_th, lr_policies_outer,
                                 lr_policies_inner):
    assert args.using_nn or args.om_using_nn

    theta_primes = copy.deepcopy(starting_th)

    f_th_primes = []

    for ii in range(n_agents):
        f_th_primes.append(higher.patch.monkeypatch(theta_primes[ii],
                                                    copy_initial_weights=True,
                                                    track_higher_grads=True))

    mixed_th_lr_policies = copy.deepcopy(lr_policies_inner)
    mixed_th_lr_policies[i] = lr_policies_outer[i]

    optims_th_primes = construct_diff_optims(theta_primes,
                                             mixed_th_lr_policies,
                                             f_th_primes)

    mixed_thetas = f_th_primes

    return mixed_thetas, optims_th_primes


if __name__ == "__main__":
    parser = argparse.ArgumentParser("NPLOLA")
    parser.add_argument("--env", type=str, default="ipd",
                        choices=["ipd"])
    parser.add_argument("--custom_param", type=str, default="random",
                        choices=['random', '1', '2', '3', '4', '5', '6', '7',
                                 'mix'],
                        help="For experimenting with custom setting for nn")
    parser.add_argument("--clip_epsilon", type=float, default=0.2,
                        help="PPO style clip hyperparameter")
    parser.add_argument("--gamma", type=float, default=0.96,
                        help="discount rate")
    parser.add_argument("--print_every", type=int, default=200,
                        help="Print every x number of epochs")
    parser.add_argument("--num_epochs", type=int, default=50001,
                        help="number of epochs to run")
    parser.add_argument("--repeats", type=int, default=1,
                        help="repeats per setting configuration")
    parser.add_argument("--n_agents_list", nargs="+", type=int, default=[5],
                        help="list of number of agents to try")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr_policies_outer", type=float, default=0.05,
                        help="outer loop learning rate: same learning rate across all policies for now")
    parser.add_argument("--lr_policies_inner", type=float, default=0.05,
                        help="inner loop learning rate (eta): this has no use in the naive learning case. Used for the gradient step done for the lookahead for other agents during LOLA (therefore, often scaled to be higher than the outer learning rate in non-proximal LOLA). Note that this has a different meaning for the Taylor approx vs. actual update versions. A value of eta=1 is perfectly reasonable for the Taylor approx version as this balances the scale of the gradient with the naive learning term (and will be multiplied by the outer learning rate after), whereas for the actual update version with neural net, 1 is way too big an inner learning rate. For prox, this is the learning rate on the inner prox loop so is not that important - you want big enough to be fast-ish, but small enough to converge.")
    parser.add_argument("--lr_values", type=float, default=0.025,
                        help="same learning rate across all policies for now. Should be around maybe 0.001 or less for neural nets to avoid instability")
    parser.add_argument("--inner_steps", type=int, default=1,
                        help="inner lookahead steps")
    parser.add_argument("--outer_steps", type=int, default=1)
    parser.add_argument("--using_nn", action="store_true",
                        help="use neural net/func approx instead of tabular policy")
    parser.add_argument("--nonlinearity", type=str, default="tanh",
                        choices=["tanh", "lrelu"])
    parser.add_argument("--nn_hidden_size", type=int, default=16)
    parser.add_argument("--nn_extra_hidden_layers", type=int, default=0)
    parser.add_argument("--set_seed", action="store_true",
                        help="set manual seed")
    parser.add_argument("--seed", type=int, default=1, help="for seed")
    parser.add_argument("--extra_value_updates", type=int, default=0,
                        help="additional value function updates (0 means just 1 update per outer rollout)")
    parser.add_argument("--history_len", type=int, default=1,
                        help="Number of steps lookback that each agent gets as state")
    parser.add_argument("--init_state_representation", type=int, default=2,
                        help="2 = separate state. 1 = coop. 0 = defect (0 not tested, recommended not to use)")
    parser.add_argument("--rollout_len", type=int, default=50,
                        help="How long we want the time horizon of the game to be (number of steps before termination/number of iterations of the IPD)")
    parser.add_argument("--base_cf_no_scale", type=float, default=1.33,
                        help="base contribution factor for no scaling (right now for 2 agents)")
    parser.add_argument("--base_cf_scale", type=float, default=0.6,
                        help="base contribution factor with scaling (right now for >2 agents)")
    parser.add_argument("--std", type=float, default=0.1,
                        help="standard deviation for initialization of policy/value parameters")
    parser.add_argument("--inner_beta", type=float, default=0,
                        help="beta determines how strong we want the KL penalty to be. Used with inner_exact_prox ")
    parser.add_argument("--outer_beta", type=float, default=0,
                        help="beta determines how strong we want the KL penalty to be. Used with outer_exact_prox ")
    parser.add_argument("--print_inner_rollouts", action="store_true")
    parser.add_argument("--inner_exact_prox", action="store_true",
                        help="find exact prox solution in inner loop instead of x # of inner steps")
    parser.add_argument("--outer_exact_prox", action="store_true",
                        help="find exact prox solution in outer loop instead of x # of outer steps")
    parser.add_argument("--actual_update", action="store_true",
                        help="experimental: try DiCE style, direct update of policy and diff through it. This is the no taylor approximation version of LOLA")
    parser.add_argument("--taylor_with_actual_update", action="store_true",
                        help="experimental: Like no taylor approx, except instead of recalc the value at outer step, calc value based on taylor approx from original value (see LOLA paper 4.2)")
    parser.add_argument("--ill_condition", action="store_true",
                        help="in exact case, add preconditioning to make the problem ill-conditioned. Used to test if prox-lola helps")
    parser.add_argument("--print_prox_loops_info", action="store_true",
                        help="print some additional info for the prox loop iters")
    parser.add_argument("--prox_threshold", type=float, default=1e-8,
                        help="Threshold for KL divergence below which we consider a fixed point to have been reached for the proximal LOLA. Recommended not to go to higher than 1e-8 if you want something resembling an actual fixed point")
    parser.add_argument("--prox_inner_max_iters", type=int, default=10000,
                        help="Maximum inner proximal steps to take before timeout")
    parser.add_argument("--prox_outer_max_iters", type=int, default=5000,
                        help="Maximum outer proximal steps to take before timeout")
    parser.add_argument("--dd_stretch_factor", type=float, default=6.,
                        help="for ill conditioning in the func approx case, stretch logit of policy in DD state by this amount")
    parser.add_argument("--all_state_stretch_factor", type=float, default=0.33,
                        help="for ill conditioning in the func approx case, stretch logit of policy in all states by this amount")
    parser.add_argument("--theta_init_mode", type=str, default="standard",
                        choices=['standard', 'tft'],
                        help="For IPD/social dilemma in the exact gradient/tabular setting, choose the policy initialization mode.")
    parser.add_argument("--exact_finite_horizon", action="store_true",
                        help="Use limited horizon (rollout_len) for the exact gradient case")
    parser.add_argument("--mnist_coop_class", type=int, default=1,
                        help="Digit class to use in place of the observation when an agent cooperates, when using MNIST state representation")
    parser.add_argument("--mnist_defect_class", type=int, default=0,
                        help="Digit class to use in place of the observation when an agent defects, when using MNIST state representation")
    parser.add_argument("--visitation_weighted_kl", action="store_true",
                        help="Weight KL term in each state of policy by the discounted visitation frequency")
    parser.add_argument("--optim", type=str, default="sgd")
    parser.add_argument("--opp_model", action="store_true",
                        help="Use Opponent Modeling")
    parser.add_argument("--opp_model_steps_per_batch", type=int, default=1,
                        help="How many steps to train opp model on each batch at the beginning of each POLA epoch")
    parser.add_argument("--opp_model_data_batches", type=int, default=100,
                        help="How many batches of data (right now from rollouts) to train opp model on")
    parser.add_argument("--om_exact", action="store_true",
                        help="only in the exact case, get an exactly correct OM (up to some threshold)")
    parser.add_argument("--om_threshold", type=float, default=1e-7,
                        help="Threshold for KL divergence below which we consider the exact OM to have converged to the real opponent policy")
    parser.add_argument("--om_lr_p", type=float, default=0.2,
                        help="learning rate for opponent modeling (imitation/supervised learning) for policy")
    parser.add_argument("--om_lr_v", type=float, default=0.001,
                        help="learning rate for opponent modeling (imitation/supervised learning) for value")
    parser.add_argument("--om_using_nn", action="store_true",
                        help="use neural net/func approx instead of tabular policy FOR THE OM")
    parser.add_argument("--om_precond", action="store_true",
                        help="use precond matrix for THE OM ONLY")
    args = parser.parse_args()

    # torch.autograd.set_detect_anomaly(True)

    if args.inner_steps > 1:
        assert args.actual_update

    init_state_representation = args.init_state_representation

    rollout_len = args.rollout_len

    if args.set_seed:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    std = args.std

    # Repeats for each hyperparam setting
    repeats = args.repeats

    if args.outer_exact_prox:
        assert args.inner_exact_prox or args.actual_update

    # if args.ill_condition and not args.using_nn:

    # ill_cond_matrix1 = torch.tensor([[1, 0, -2., 0, 0.],
    #                                  [0., 1, -2, 0., 0.],
    #                                  [0, 0., 1, 0, 0.],
    #                                  [0, 0., -2., 1, 0.],
    #                                  [0, 0., -2., 0., 1.]])
    # ill_cond_matrix2 = torch.tensor([[1, -2, 0., 0, 0.],
    #                                  [0., 1, 0, 0., 0.],
    #                                  [0, -2., 1, 0, 0.],
    #                                  [0, -2., 0., 1, 0.],
    #                                  [0, -2., 0., 0., 1.]])

    ill_cond_matrix1 = torch.tensor([[1, 0, 0., 0, 0.],
                                     [-2., 1, 0, 0., 0.],
                                     [-2, 0., 1, 0, 0.],
                                     [-2, 0., 0., 1., 0.],
                                     [-2, 0., 0., 0., 1.]])
    ill_cond_matrix2 = torch.tensor([[1, 0, 0., 0, 0.],
                                     [-2., 1, 0, 0., 0.],
                                     [-2, 0., 1, 0, 0.],
                                     [-2, 0., 0., 1., 0.],
                                     [-2, 0., 0., 0., 1.]])

    ill_cond_matrices = torch.stack(
        (ill_cond_matrix1, ill_cond_matrix2))  # hardcoded 2 agents for now

    if args.ill_condition and not args.using_nn:
        print(ill_cond_matrices[0])
        print(ill_cond_matrices[1])

    if args.om_using_nn and args.om_precond:
        raise NotImplementedError("Precond only supported for tabular om policies")

    # For each repeat/run:
    num_epochs = args.num_epochs
    print_every = args.print_every
    batch_size = args.batch_size
    # Bigger batch is a big part of convergence with DiCE

    gamma = args.gamma

    if args.history_len > 1:
        assert args.using_nn  # Right now only supported for func approx.

    n_agents_list = args.n_agents_list

    if args.env != "ipd":
        raise NotImplementedError(
            "No exact gradient calcs done for this env yet")

    for n_agents in n_agents_list:

        if args.ill_condition:
            assert n_agents == 2  # Other agents not yet supported for this
            assert args.history_len == 1  # longer hist len not yet supported

        start = timer()

        assert n_agents >= 2
        if n_agents == 2:
            contribution_factor = args.base_cf_no_scale  # 1.6
            contribution_scale = False
        else:
            contribution_factor = args.base_cf_scale  # 0.6
            contribution_scale = True

        if batch_size == n_agents or batch_size == rollout_len or rollout_len == n_agents:
            raise Exception(
                "Having two of batch size, rollout len, or n_agents equal will cause insidious bugs when reshaping dimensions")
            # I should really refactor/think about a way to avoid these bugs. Right now this is kind of a meh patch

        lr_policies_outer = torch.tensor([args.lr_policies_outer] * n_agents)
        lr_policies_inner = torch.tensor([args.lr_policies_inner] * n_agents)

        lr_values = torch.tensor([args.lr_values] * n_agents)

        if not contribution_scale:
            inf_coop_payout = 1 / (1 - gamma) * (contribution_factor - 1)
            truncated_coop_payout = inf_coop_payout * \
                                    (
                                            1 - gamma ** rollout_len)  # This last term here accounts for the fact that we don't go to infinity
            inf_max_payout = 1 / (1 - gamma) * (
                    contribution_factor * (n_agents - 1) / n_agents)
            truncated_max_payout = inf_max_payout * \
                                   (1 - gamma ** rollout_len)
        else:
            inf_coop_payout = 1 / (1 - gamma) * (
                        contribution_factor * n_agents - 1)
            truncated_coop_payout = inf_coop_payout * \
                                    (1 - gamma ** rollout_len)
            inf_max_payout = 1 / (1 - gamma) * (
                        contribution_factor * (n_agents - 1))
            truncated_max_payout = inf_max_payout * \
                                   (1 - gamma ** rollout_len)

        max_single_step_return = (
                    contribution_factor * (n_agents - 1) / n_agents)

        adjustment_to_make_rewards_negative = 0
        # adjustment_to_make_rewards_negative = max_single_step_return

        discounted_sum_of_adjustments = 1 / (
                1 - gamma) * adjustment_to_make_rewards_negative * \
                                        (1 - gamma ** rollout_len)

        print("Number of agents: {}".format(n_agents))
        print("Contribution factor: {}".format(contribution_factor))
        print("Scaled contribution factor? {}".format(contribution_scale))

        print("Exact Gradients")

        if args.actual_update:
            print("Actual Update LOLA")
        else:
            print("Original LOLA")

        num_found_tft = 0

        pol0_record = []
        pol1_record = []

        for run in range(repeats):

            if args.env == "ipd":
                game = ContributionGame(n=n_agents, gamma=gamma,
                                        batch_size=batch_size,
                                        num_iters=rollout_len,
                                        contribution_factor=contribution_factor,
                                        contribution_scale=contribution_scale,
                                        history_len=args.history_len)
                dims = game.dims
                if args.opp_model:
                    om_dims = game.om_dims
            else:
                raise NotImplementedError

            th = init_custom(dims, args.using_nn,
                             args.nn_hidden_size, args.nn_extra_hidden_layers)

            opp_models = None
            if args.opp_model:
                opp_models = []
                for i in range(n_agents):
                    # Including an opp model of self which is never used
                    agent_i_opp_models = init_custom(om_dims, args.om_using_nn,
                                 args.nn_hidden_size, args.nn_extra_hidden_layers)
                    # print(agent_i_opp_models)
                    opp_models.append(agent_i_opp_models)

            # Run
            G_ts_record = torch.zeros((num_epochs, n_agents))

            tft_found = False

            for epoch in range(num_epochs):
                if epoch == 0:
                    print("lr_policies_outer: {}".format(lr_policies_outer))
                    print("lr_policies_inner: {}".format(lr_policies_inner))
                    print("Starting Policies:")
                    game.print_policies_for_all_states(th, ill_cond=args.ill_condition)

                # print("before OM")
                # for i in range(len(th)):
                #     game.print_policies_for_all_states(opp_models[i])

                if args.opp_model:
                    if args.om_exact:
                        for i in range(len(th)):
                            for j in range(len(th)):
                                if i != j:
                                    # TODO CHECK THIS GENERALIZES CORRECTLY TO N>2 AGENTS
                                    kl_div = 9999
                                    om_iters = 0
                                    om_max_iters = 10000
                                    while kl_div > args.om_threshold and om_iters < om_max_iters:
                                        # Learn the OM using exact probabilities in each state
                                        kl_div = game.learn_om_from_policy(th, opp_models, i, j)
                                        om_iters += 1
                                    # print(f"OM Iters used: {om_iters}")

                    else:
                        # DO fixed number of iterations
                        raise NotImplementedError

                # if args.opp_model:
                #     print("After OM")
                #     for x in range(len(th)):
                #         game.print_policies_for_all_states(opp_models[x], ill_cond=args.om_precond)



                if args.actual_update:
                    th = update_th_exact_value(th, game, opp_models=opp_models)
                else:
                    th = update_th_taylor_approx_exact_value(th, game)

                # Reevaluate to get the G_ts from synchronous play
                losses = game.get_exact_loss(th, ill_cond=args.ill_condition)
                G_ts_record[epoch] = -torch.stack(losses).detach()

                if (epoch + 1) % print_every == 0:
                    print("Epoch: " + str(epoch + 1), flush=True)
                    curr = timer()
                    print("Time Elapsed: {:.1f} seconds".format(curr - start))

                    print("Discounted Rewards: {}".format(G_ts_record[epoch]))
                    print("Max Avg Coop Payout (Infinite Horizon): {:.3f}".format(
                            inf_coop_payout))

                    game.print_policies_for_all_states(th, ill_cond=args.ill_condition)

                    # print("OMs")
                    # for i in range(len(th)):
                    #     game.print_policies_for_all_states(opp_models[i], ill_cond=args.om_precond)

                # at_least_one_tft = False
                scores = G_ts_record[epoch]
                policy0 = game.get_policy_for_all_states(th, 0, ill_cond=args.ill_condition)
                policy1 = game.get_policy_for_all_states(th, 1, ill_cond=args.ill_condition)
                tft_coop_threshold = 0.65
                tft_return_threshold = 0.8
                at_least_one_tft = (policy0[0] < tft_coop_threshold and policy0[2] < tft_coop_threshold) or (policy1[0] < tft_coop_threshold and policy1[1] < tft_coop_threshold)
                both_tft = (policy0[0] < tft_coop_threshold and policy0[2] < tft_coop_threshold) and (policy1[0] < tft_coop_threshold and policy1[1] < tft_coop_threshold)
                if scores.mean() > (inf_coop_payout * tft_return_threshold) and both_tft:
                    tft_found = True

            if tft_found:
                print("TFT Found")
                num_found_tft += 1
            else:
                print("TFT Not Found")

            pol0 = game.get_policy_for_all_states(th, 0)
            pol1 = game.get_policy_for_all_states(th, 1)

            pol0_record.append(pol0)
            pol1_record.append(pol1)

            # % comparison of average individual reward to max average individual reward
            # This gives us a rough idea of how close to optimal (how close to full cooperation) we are.
            # But may not be ideal. Metrics like how often TFT is found eventually (e.g. within x epochs)
            # may be more valuable/more useful for understanding and judging.
            if args.exact_finite_horizon:
                print(
                    "Warning: finite horizon not well tested. May need code modification")
                coop_divisor = truncated_coop_payout
            else:
                coop_divisor = inf_coop_payout

            plot_results = True
            if plot_results:
                now = datetime.datetime.now()

                avg_gts_to_plot = G_ts_record
                plt.plot(avg_gts_to_plot)

                plt.savefig(
                    "{}agents_outerlr{}_innerlr{}_run{}_exact_date{}.png".format(
                        n_agents, args.lr_policies_outer,
                        args.lr_policies_inner, run,
                        now.strftime('%Y-%m-%d_%H-%M')))

                plt.clf()

        pol0_record = torch.stack((pol0_record)).mean(dim=0)
        pol1_record = torch.stack((pol1_record)).mean(dim=0)
        avg_over_agents = pol0_record
        avg_over_agents[0] += pol1_record[0]
        avg_over_agents[1] += pol1_record[2]
        avg_over_agents[2] += pol1_record[1]
        avg_over_agents[3] += pol1_record[3]
        avg_over_agents[4] += pol1_record[4]
        avg_over_agents /= 2.


        def print_special(pol_rec):
            print(
                f"& {pol_rec[0]:.2f} & {pol_rec[1]:.2f} & {pol_rec[2]:.2f} & {pol_rec[3]:.2f} & {pol_rec[4]:.2f} \\\\")


        print_special(avg_over_agents)

        print(f"Num Runs Where TFT Found: {num_found_tft}")
        print(f"Proportion of Runs Where TFT Found: {num_found_tft / repeats}")
