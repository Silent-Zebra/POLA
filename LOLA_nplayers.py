import numpy as np
import torch
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F

# import higher

import datetime

import copy

import argparse

init_state_representation = 2  # Change here if you want different number to represent the initial state
rollout_len = 50

theta_init_modes = ['standard', 'tft']
theta_init_mode = 'standard'

def bin_inttensor_from_int(x, n):
    return torch.Tensor([int(d) for d in (str(bin(x))[2:]).zfill(n)])
    # return [int(d) for d in str(bin(x))[2:]]

# https://stackoverflow.com/questions/55918468/convert-integer-to-pytorch-tensor-of-binary-bits
# def int_from_bin_inttensor(bin_tens):
#     bits = len(bin_tens)
#     # mask = 2 ** torch.arange(bits - 1, -1, -1).to(bin_tens.device, bin_tens.dtype)
#     mask = 2 ** torch.arange(bits - 1, -1, -1)
#
#     # print(mask)
#
#     # index = 0
#     # for i in range(len(bin_tens)):
#     #     index += 2**i * bin_tens[-i-1]
#     #
#     # print(int(index.item()))
#     # print(torch.sum(mask * bin_tens, -1))
#     # 1/0
#
#     return torch.sum(mask * bin_tens, -1)
#
# def int_from_bin_inttensor(dec_values_mask, bin_tens):
#     return torch.sum(dec_values_mask * bin_tens, -1)

def build_bin_matrix(n, size):
    bin_mat = torch.zeros((size, n))
    for i in range(size):
        l = bin_inttensor_from_int(i, n)
        bin_mat[i] = l
    return bin_mat


def build_p_vector(n, size, pc, bin_mat):
    pc = pc.repeat(size).reshape(size, n)
    pd = 1 - pc
    # p = torch.zeros(size)
    p = torch.prod(bin_mat * pc + (1 - bin_mat) * pd, dim=1)
    return p


def magic_box(x):
    return torch.exp(x - x.detach())


def copyNN(target_net, curr_net):
    # Copy from curr to target
    target_net.load_state_dict(curr_net.state_dict())

def optim_update(diffopt, loss, params):
    return diffopt.step(loss, params)
    # optim.zero_grad()
    # loss.backward(retain_graph=True)
    # optim.step()

# def copy_thetas(th):
#     # th_copy = []
#     # for i in range(len(th)):
#     #     if isinstance(th[i], NeuralNet):
#     #         th_copy.append()
#     return copy.deepcopy(th)
def reverse_cumsum(x, dim):
    return x + torch.sum(x, dim=dim, keepdims=True) - torch.cumsum(x, dim=dim)

def ipdn(n=2, gamma=0.96, contribution_factor=1.6, contribution_scale=False):
    dims = [2 ** n + 1 for _ in range(n)]
    state_space = dims[0]
    # print(dims)

    if contribution_scale:
        contribution_factor = contribution_factor * n
    else:
        assert contribution_factor > 1
    # contribution_factor = 1.7
    # contribution_factor = 0.6 * n

    bin_mat = build_bin_matrix(n, size=state_space - 1)

    payout_vectors = torch.zeros((n,
                                  state_space - 1))  # one vector for each player, each player has n dim vector for payouts in each of the n states
    for agent in range(n):
        for state in range(state_space - 1):
            l = bin_inttensor_from_int(state, n)
            total_contrib = sum(l)
            agent_payout = total_contrib * contribution_factor / n - l[
                agent]  # if agent contributed 1, subtract 1
            agent_payout -= adjustment_to_make_rewards_negative
            payout_vectors[agent][state] = agent_payout

    def Ls(th):

        # Theta denotes (unnormalized) action probabilities at each of the states:
        # start CC CD DC DD

        init_pc = torch.zeros(n)
        for i in range(n):
            # p_i_0 = torch.sigmoid(th[i][0:1])
            p_i_0 = torch.sigmoid(th[i][
                                      -1])  # so start state is at the end, different from the 2p ipd formulation
            # Why did I do it this way? It's inconsistent with the 2p ipd setup
            # print(torch.sigmoid(th[i][-1]))
            init_pc[i] = p_i_0
            # Anyway policy still represents prob of coop (taking action 1)

        # Here's what we'll do for the state representation
        # binary number which increments
        # and then for 1 you can take the coop prob and 0 you can take the defect prob
        # So 111111...111 is all coop
        # and 000...000 is all defect
        # Then the last state which is 1000...000 is the start state
        # So we're kinda working backwards here... CCCC...CCC is the second last table/matrix/vector entry

        p = build_p_vector(n, state_space - 1, init_pc, bin_mat)
        # p = build_p_vector(n=n, size=state_space-1, pc=init_pc)

        # TODO this part can almost certainly be optimized
        # Probabilities in the states other than the start state
        all_p_is = torch.zeros((n, state_space - 1))
        for i in range(n):
            p_i = torch.sigmoid(th[i][0:-1])
            # p_i = torch.reshape(torch.sigmoid(th[i][0:-1]), (-1, 1)) # or just -1 instead of -1,1
            all_p_is[i] = p_i
        # print(all_p_is.shape)

        # TODO is there a way to optimize this part and remove the loop?
        # Transition Matrix
        # Remember now our transition matrix top left is DDD...D to DDD...D
        # 0 is defect in this formulation, 1 is contributing a resource value of 1
        # if you want to think of it that way
        P = torch.zeros((state_space - 1, state_space - 1))
        for curr_state in range(state_space - 1):
            i = curr_state
            # pc = all_p_is[:, i, :]
            pc = all_p_is[:, i]
            p_new = build_p_vector(n, state_space - 1, pc, bin_mat)
            # p_new = build_p_vector(n, state_space-1, pc)
            P[i] = p_new

        M = torch.matmul(p,
                         torch.inverse(torch.eye(state_space - 1) - gamma * P))

        # Remember M is just the steady state probabilities for each of the states
        # It is a vector, not a matrix.

        L_all = []
        for i in range(n):
            payout_vec = payout_vectors[i]
            L_i = torch.matmul(M, payout_vec)
            L_all.append(L_i)

        return L_all

    return dims, Ls

class ContributionGame():
    def __init__(self, n, batch_size, num_iters, gamma=0.96, contribution_factor=1.6, contribution_scale=False):
        self.n_agents = n
        self.gamma = gamma
        self.contribution_factor = contribution_factor
        self.contribution_scale = contribution_scale
        self.dims = [n] * n
        self.batch_size = batch_size
        self.num_iters = num_iters

        if contribution_scale:
            self.contribution_factor = contribution_factor * n
        else:
            assert self.contribution_factor > 1

        self.dec_value_mask = (2 ** torch.arange(n - 1, -1, -1)).float()

    def get_init_state_batch(self):
        init_state_batch = torch.ones(
            (self.batch_size, self.n_agents)) * init_state_representation
        return init_state_batch

    def int_from_bin_inttensor(self, bin_tens):

        return torch.sum(self.dec_value_mask * bin_tens, -1).item()

    def get_state_batch_indices(self, state_batch, iter):
        if iter == 0:
            # we just started
            assert state_batch[0][0] - init_state_representation == 0
            indices = [-1] * self.batch_size
        else:
            indices = list(map(self.int_from_bin_inttensor, state_batch))
        return indices

    def get_policy_and_state_value(self, pol, val, state_batch, state_batch_indices=None):
        if isinstance(pol, torch.Tensor):
            assert state_batch_indices is not None

            # if iter == 0:
            #     assert state_batch[0][0] - init_state_representation == 0
            #     indices = [-1] * self.batch_size
            #
            # else:
            #     indices = list(map(int_from_bin_inttensor, state_batch))

            # policy = torch.sigmoid(pol)[indices].reshape(-1, 1)
            #
            # state_value = val[indices].reshape(-1, 1)

            # print(state_batch_indices)

            policy = torch.sigmoid(pol)[state_batch_indices].reshape(-1, 1)

            state_value = val[state_batch_indices].reshape(-1, 1)

        else:
            # policy = th[i](state)
            policy = pol(state_batch)
            state_value = val(state_batch)

        return policy, state_value

    def get_policy_vals_indices_for_iter(self, th, vals, state_batch, iter):
        policies = torch.zeros((self.n_agents, self.batch_size, 1))
        state_values = torch.zeros((self.n_agents, self.batch_size, 1))
        state_batch_indices = self.get_state_batch_indices(state_batch,
                                                           iter)
        for i in range(self.n_agents):

            policy, state_value = self.get_policy_and_state_value(th[i],
                                                                  vals[i],
                                                                  state_batch,
                                                                  state_batch_indices)

            policies[i] = policy
            state_values[i] = state_value

        return policies, state_values, state_batch_indices

    def get_next_val_history(self, th, vals, val_history, ending_state_batch, iter):

        policies, ending_state_values, ending_state_indices = self.get_policy_vals_indices_for_iter(
            th, vals, ending_state_batch, iter)

        # ending_state_values = torch.zeros((self.n_agents, self.batch_size, 1))
        # for i in range(self.n_agents):
        #     policy, state_value = self.get_policy_and_state_value(th[i],
        #                                                           vals[i],
        #                                                           state_batch,
        #                                                           state_batch_indices)
        #     ending_state_values[i] = state_value

        next_val_history = torch.zeros(
            (self.num_iters, self.n_agents, self.batch_size, 1))
        next_val_history[:self.num_iters - 1, :, :, :] = \
            val_history[1:self.num_iters, :, :, :]
        next_val_history[-1, :, :, :] = ending_state_values

        return next_val_history

    def get_policies_vals_for_states(self, th, vals, trajectory):
        # Returns coop_probs and state_vals, which are the equivalent of
        # policy_history and val_history, except they are using the current policies and values
        # (which may be different from the policies and values that were originally
        # used to rollout in the environment)

        # TODO batch this, to be faster and avoid for loops
        # Look through entire code for for loops
        # Finally, move everything to GPU? And then test that also.

        coop_probs = torch.zeros((self.num_iters, self.n_agents, self.batch_size, 1))
        init_state_batch = self.get_init_state_batch()
        state_batch = init_state_batch

        state_vals = torch.zeros(
            (self.num_iters, self.n_agents, self.batch_size, 1))

        # trajectory_indices = self.get_state_batch_indices(state_batch,
        #                                                        iter)
        # for i in range(self.n_agents):
        #     policy, state_value = self.get_policy_and_state_value(th[i], vals[i],
        #                                                           trajectory,
        #                                                           trajectory_indices)

        for iter in range(self.num_iters):
            # policies = torch.zeros((self.n_agents, self.batch_size, 1))
            # state_values = torch.zeros((self.n_agents, self.batch_size, 1))
            # state_batch_indices = self.get_state_batch_indices(state_batch,
            #                                                    iter)
            #
            # for i in range(self.n_agents):
            #
            #     policy, state_value = self.get_policy_and_state_value(th[i], vals[i], state_batch, state_batch_indices)
            #
            #     policies[i] = policy
            #     state_values[i] = state_value

            policies, state_values, state_batch_indices = self.get_policy_vals_indices_for_iter(
                th, vals, state_batch, iter)

            coop_probs[iter] = policies
            state_vals[iter] = state_values

            state_batch = trajectory[iter] # get the next state batch from the trajectory
            state_batch = state_batch.reshape(self.n_agents, self.batch_size)
            state_batch = state_batch.t()
        next_val_history = self.get_next_val_history(th, vals, state_vals, state_batch,
                                                     iter + 1)

        return coop_probs, state_vals, next_val_history


    def rollout(self, th, vals):
        # init_state = torch.Tensor([[init_state_representation] * self.n_agents])  # repeat -1 n times, where n is num agents
        # print(init_state)
        init_state_batch = self.get_init_state_batch()
        # print(init_state_batch)

        # state = init_state
        state_batch = init_state_batch

        # trajectory just tracks actions, doesn't track the init state
        trajectory = torch.zeros((self.num_iters, self.n_agents, self.batch_size, 1), dtype=torch.int)
        rewards = torch.zeros((self.num_iters, self.n_agents, self.batch_size, 1))
        policy_history = torch.zeros((self.num_iters, self.n_agents, self.batch_size, 1))
        val_history = torch.zeros((self.num_iters, self.n_agents, self.batch_size, 1))

        # This loop can't be skipped due to sequential nature of environment
        for iter in range(self.num_iters):

            # policies = torch.zeros((self.n_agents, self.batch_size, 1))
            # state_values = torch.zeros((self.n_agents, self.batch_size, 1))
            #
            # state_batch_indices = self.get_state_batch_indices(state_batch, iter)
            # for i in range(self.n_agents):
            #     # This loop can't be skipped unless we assume all of the thetas follow the same structure (e.g. tensor)
            #
            #     policy, state_value = self.get_policy_and_state_value(th[i], vals[i], state_batch, state_batch_indices)
            #
            #     # policy prob of coop between 0 and 1
            #     policies[i] = policy
            #     state_values[i] = state_value

            policies, state_values, state_batch_indices = self.get_policy_vals_indices_for_iter(th, vals, state_batch, iter)

            policy_history[iter] = policies
            val_history[iter] = state_values

            actions = torch.distributions.binomial.Binomial(probs=policies.detach()).sample()

            state_batch = torch.Tensor(actions)
            state_batch = state_batch.reshape(self.n_agents, self.batch_size)
            state_batch = state_batch.t()

            trajectory[iter] = torch.Tensor(actions)

            # Note negative rewards might help with exploration in PG formulation
            total_contrib = sum(actions)
            # total_contrib = actions.sum(dim=0)

            payout_per_agent = total_contrib * self.contribution_factor / self.n_agents
            agent_rewards = -actions + payout_per_agent  # if agent contributed 1, subtract 1, that's what the -actions does
            agent_rewards -= adjustment_to_make_rewards_negative
            rewards[iter] = agent_rewards

            # print(actions)
            # print(payout_per_agent)
            # print(agent_rewards)

        # ending_state_values = torch.zeros((self.n_agents, self.batch_size, 1))
        # for i in range(self.n_agents):
        #     policy, state_value = self.get_policy_and_state_value(th[i], vals[i],
        #                                                           state_batch,
        #                                                           state_batch_indices)
        #     ending_state_values[i] = state_value
        #
        # next_val_history = torch.zeros((self.num_iters, self.n_agents, self.batch_size, 1))
        # next_val_history[:self.num_iters-1,:,:,:] = val_history[1:self.num_iters,:,:,:]
        # next_val_history[-1,:,:,:] = ending_state_values
        next_val_history = self.get_next_val_history(th, vals, val_history, state_batch, iter + 1) # iter doesn't even matter here as long as > 0

        return trajectory, rewards, policy_history, val_history, next_val_history

    def get_loss_helper(self, trajectory, rewards, policy_history, old_policy_history = None):
        num_iters = len(trajectory)

        discounts = torch.cumprod(self.gamma * torch.ones((num_iters)),
                                  dim=0) / self.gamma

        # Discounted rewards, not sum of rewards which G_t is
        gamma_t_r_ts = rewards * discounts.reshape(-1, 1, 1,
                                                   1)  # implicit broadcasting done by numpy

        G_ts = reverse_cumsum(gamma_t_r_ts, dim=0)
        # G_ts = reverse_cumsum(rewards * discounts.reshape(-1, 1, 1, 1), dim=0)
        # G_ts gives you the inner sum of discounted rewards


        p_act_given_state = trajectory.float() * policy_history + (
                1 - trajectory.float()) * (1 - policy_history)


        if old_policy_history is None:

            # recall 1 is coop, so when coop action 1 taken, we look at policy which is prob coop
            # and when defect 0 is taken, we take 1-policy = prob of defect

            log_p_act = torch.log(p_act_given_state)
            return G_ts, gamma_t_r_ts, log_p_act, discounts
        else:
            p_act_given_state_old = trajectory.float() * old_policy_history + (
                    1 - trajectory.float()) * (1 - old_policy_history)

            p_act_ratio = p_act_given_state / p_act_given_state_old.detach()
            # TODO is this detach necessary?


            return G_ts, gamma_t_r_ts, p_act_ratio, discounts




    def get_gradient_terms(self, trajectory, rewards, policy_history):

        G_ts, gamma_t_r_ts, log_p_act = self.get_loss_helper(trajectory,
                                                             rewards,
                                                             policy_history)

        # These are basically grad_i E[R_0^i] - naive learning loss
        # no LOLA loss here yet
        # objective_nl = (torch.cumsum(log_p_act, dim=0) * gamma_t_r_ts).sum(
        #     dim=0)
        objective_nl = (log_p_act * G_ts).sum(dim=0)
        # losses_nl = (log_p_act * G_ts).sum(dim=0)

        log_p_times_G_t_matrix = torch.zeros((self.n_agents, self.n_agents))
        # so entry 0,0 is - (log_p_act[:,0] * G_ts[:,0]).sum(dim=0)
        # entry 1,1 is - (log_p_act[:,1] * G_ts[:,1]).sum(dim=0)
        # and so on
        # entry 0,1 is - (log_p_act[:,0] * G_ts[:,1]).sum(dim=0)
        # and so on
        # Be careful with dimensions/not to mix them up


        for i in range(self.n_agents):
            for j in range(self.n_agents):
                # print(log_p_act[:, i].shape)
                # print(G_ts[:, j].shape)
                # print((log_p_act[:, i] * G_ts[:, j]).sum(dim=0))

                # print((log_p_act[:, i] * G_ts[:, j]).shape)
                # print((log_p_act[:, i] * G_ts[:, j]).sum(dim=0).shape)
                # print((log_p_act[:, i] * G_ts[:, j]).sum(dim=0).mean(dim=0).shape)


                log_p_times_G_t_matrix[i][j] = (
                            log_p_act[:, i] * G_ts[:, j]).sum(dim=0).mean(dim=0)
        # Remember that the grad corresponds to the log_p and the R_t corresponds to the G_t
        # We can switch the log_p and G_t (swap log_p i to j and vice versa) if we want to change order

        # For the first term, my own derivation showed that
        # grad_2 E(R_1) = (prop to) Sum (grad_2 (log pi_2)) G_t(1)

        # For the grad_1 grad_2 term:
        # the way this will work is that the ith entry (row) in this log_p_act_sums_0_to_t
        # will be the sum of log probs from time 0 to time i
        # Then the dimension of each row is the number of agents - we have the sum of log probs
        # for each agent
        # later we will product them (but in pairwise combinations!)

        log_p_act_sums_0_to_t = torch.cumsum(log_p_act, dim=0)

        # Remember also that for p1 you want grad_1 grad_2 of R_2 (P2's return)
        # So then you also want grad_1 grad_3 of R_3
        # and so on

        grad_1_grad_2_matrix = torch.zeros((self.n_agents, self.n_agents, self.batch_size, 1))
        for i in range(self.n_agents):
            for j in range(self.n_agents):
                grad_1_grad_2_matrix[i][j] = (torch.FloatTensor(gamma_t_r_ts)[:,
                                              j] * log_p_act_sums_0_to_t[:,
                                                   i] * log_p_act_sums_0_to_t[:,
                                                        j]).sum(dim=0)
        # Here entry i j is grad_i grad_j E[R_j]

        # TODO NOTE THESE ARE NOT LOSSES, THEY ARE REWARDS (discounted)
        # Need to negative if you will torch optim on them. This is a big issue lol
        # objective = objective_nl

        grad_log_p_act = []

        for i in range(self.n_agents):
            # TODO could probably get this without taking grad, could be more efficient
            # print(log_p_act[0, i].shape)

            # CANNOT DO mean here. Must do individually for every batch, preserving the batch_size dimension
            # until later.
            # Once done, then test with just 2 players that you get the right result.

            example_grad = get_gradient(log_p_act[0, i, 0], th[i]) if isinstance(
                th[i], torch.Tensor) else torch.cat(
                [get_gradient(log_p_act[0, i, 0], param).flatten() for
                 param in
                 th[i].parameters()])
            grad_len = len(example_grad)
            grad_log_p_act.append(torch.zeros((rollout_len, self.batch_size, grad_len)))

        for i in range(self.n_agents):

            for t in range(rollout_len):

                for b in range(self.batch_size):

                    grad_t = get_gradient(log_p_act[t, i, b], th[i]) if isinstance(
                        th[i], torch.Tensor) else torch.cat(
                        [get_gradient(log_p_act[t, i, b], param).flatten() for
                         param in
                         th[i].parameters()])


                    grad_log_p_act[i][t][b] = grad_t

        return objective_nl, grad_1_grad_2_matrix, log_p_times_G_t_matrix, G_ts, gamma_t_r_ts, log_p_act_sums_0_to_t, log_p_act, grad_log_p_act

    def get_dice_loss(self, trajectory, rewards, policy_history, val_history, next_val_history, old_policy_history=None):

        G_ts, gamma_t_r_ts, log_p_act_or_p_act_ratio, discounts = self.get_loss_helper(
            trajectory, rewards, policy_history, old_policy_history)

        # print(log_p_act_or_p_act_ratio)
        # if old_policy_history is None:
        #     G_ts, gamma_t_r_ts, log_p_act, discounts = self.get_loss_helper(trajectory, rewards, policy_history)
        # else:
        #     G_ts, gamma_t_r_ts, p_act_ratio, discounts = self.get_loss_helper(
        #         trajectory, rewards, policy_history, old_policy_history)

        discounts = discounts.view(-1, 1, 1, 1)

        discounted_vals = val_history * discounts

        # R_t is like G_t except not discounted back to the start. It is the forward
        # looking return at that point in time
        R_ts = G_ts / discounts

        # print(R_ts[1])
        # print(G_ts[1])

        # advantages = G_ts - discounted_vals
        advantages = rewards + self.gamma * next_val_history - val_history
        # print(advantages)


        # TODO Aug 24 - make sure the basic version without the clipping stuff or the inner loops is working properly
        # Something isn't right... figure out why the basic version is not working, compare vs older versions
        # Start with the = false flag on the repeat_train_on_samples and ensure that is working first before testing with true flag.
        # and do a step by step modification/comparison and test of all different things to see what the issue is


        if repeat_train_on_same_samples and use_clipping:

            # PPO style clipping

            probs_to_clip = (advantages > 0).float()

            log_p_act_or_p_act_ratio = probs_to_clip * torch.minimum(log_p_act_or_p_act_ratio,torch.zeros_like(log_p_act_or_p_act_ratio) + 1+clip_epsilon) + (1-probs_to_clip) * log_p_act_or_p_act_ratio

        # if repeat_train_on_same_samples and use_clipping:
        #
        #     # print(log_p_act_or_p_act_ratio)
        #
        #     # full clipping - sometimes in rare cases the not clipping on the negative side leads to way too big a movement and then you have essentially
        #     # a stuck policy (e.g. too close to 0 or 1 for any hope of future exploration) which is always bad
        #     log_p_act_or_p_act_ratio = torch.clamp(log_p_act_or_p_act_ratio, min=1 - clip_epsilon, max=1 + clip_epsilon)

        # print(log_p_act_or_p_act_ratio)



        # Find the indices/probs where the advantage is negative for each player
        # Then for each of those, we do nothing
        # THen for the other probs where the advantage is positive
        # we clip those (so we really only need the min/floor/clamp on the top part to make sure we don't move too much in that direction)
        # TODO Aug 23 confirm that this is an appropriate interpretation of the PPO clipping.

        # p_act_ratio = torch.clamp(p_act_ratio, min=1 - clip_epsilon,
        #                           max=1 + clip_epsilon)

        sum_over_agents_log_p_act_or_p_act_ratio = log_p_act_or_p_act_ratio.sum(dim=1)

        # See 5.2 (page 7) of DiCE paper for below:
        # With batches, the mean is the mean across batches. The sum is over the steps in the rollout/trajectory

        # a=torch.cumsum(sum_over_agents_log_p_act_or_p_act_ratio, dim=0).reshape(-1, 1, self.batch_size, 1) * gamma_t_r_ts
        #
        # b = sum_over_agents_log_p_act_or_p_act_ratio.reshape(-1, 1, self.batch_size, 1) * G_ts
        #
        # a = a.sum(dim=0).mean(dim=1)
        # b = b.sum(dim=0).mean(dim=1)
        #
        # print(a)
        # print(b)
        #
        # print(a-b)

        deps_up_to_t = (torch.cumsum(sum_over_agents_log_p_act_or_p_act_ratio, dim=0)).reshape(-1, 1, self.batch_size, 1)

        deps_less_than_t = deps_up_to_t - sum_over_agents_log_p_act_or_p_act_ratio.reshape(-1, 1, self.batch_size, 1) # take out the dependency in the given time step

        # Two equivalent formulations - well they should be but somehow the G_t one is wrong.
        # dice_rewards2 is wrong because while the first order gradients match, the higher order ones don't.
        # Look at how the magic box operator works and think about why what was equivalent
        # formulations in the regular policy gradient case is no longer equivalent with the magic box on it
        dice_rewards = (magic_box(torch.cumsum(sum_over_agents_log_p_act_or_p_act_ratio, dim=0)).reshape(-1, 1, self.batch_size, 1) * gamma_t_r_ts).sum(dim=0).mean(dim=1)
        dice_rewards2 = (magic_box(sum_over_agents_log_p_act_or_p_act_ratio.reshape(-1, 1, self.batch_size, 1)) * G_ts).sum(dim=0).mean(dim=1)
        # loaded_dice_rewards = ((magic_box(deps_up_to_t) - magic_box(deps_less_than_t)) * discounts * R_ts).sum(dim=0).mean(dim=1)
        # loaded_dice_rewards = ((magic_box(deps_up_to_t) - magic_box(deps_less_than_t)) * G_ts).sum(dim=0).mean(dim=1)

        loaded_dice_rewards = ((magic_box(deps_up_to_t) - magic_box(deps_less_than_t)) * discounts * advantages).sum(dim=0).mean(dim=1)


        # print(dice_rewards)
        # print(dice_rewards2)
        # dice_rewards = dice_rewards2

        # dice_objective_w_baseline = (magic_box(sum_over_agents_log_p_act_or_p_act_ratio.reshape(-1, 1, self.batch_size, 1)) * advantages).sum(dim=0).mean(dim=1)

        # Each state has a baseline. For each state, the sum of nodes w are all of the log probs
        # So why is the G_t reward formulation wrong? How is that different from the baseline formulation?
        # baseline_term = ((1 - magic_box(sum_over_agents_log_p_act_or_p_act_ratio.reshape(-1, 1, self.batch_size, 1))) * discounted_vals).sum(dim=0).mean(dim=1)

        # dice_objective_w_baseline = dice_rewards + baseline_term
        # dice_objective_w_baseline = loaded_dice_rewards + baseline_term

        # dice_loss = -dice_objective_w_baseline
        dice_loss = -loaded_dice_rewards

        # values_loss = ((G_ts - discounted_vals) ** 2).sum(dim=0).mean(dim=1)

        # values_loss = ((rewards + gamma * next_val_history - val_history) ** 2).sum(dim=0).mean(dim=1)

        final_state_vals = next_val_history[-1]
        # final_state_vals = next_val_history[-1].detach()
        # Detach would be wrong here I think
        values_loss = ((R_ts + (gamma * discounts.flip(dims=[0])) * final_state_vals.reshape(1, *final_state_vals.shape) - val_history) ** 2).sum(dim=0).mean(dim=1)
        # print((gamma * discounts.flip(dims=[0])) * final_state_vals.reshape(1, *final_state_vals.shape))
        # print(R_ts + (gamma * discounts.flip(dims=[0])) * final_state_vals.reshape(1, *final_state_vals.shape))
        # print(val_history)


        # Hm so I think this might be interesting. By choosing discounted_vals as the term for the baseline
        # No that's definitely correct, you definitely need that since you are comparing against G_t
        # But what's interesting is that in terms of the loss for the value function,
        # I should be using G_ts - discounted_vals, but brought forward so that discounting starts from 0
        # and suppose the true value from start is ~13, then the value in each state should be closer to 7 or so depending
        # on the discounting - because in the last time step, the value is pretty close to 0
        # So there is an issue resulting from the fixed time step termination when we should be doing probabilistic termination
        # But anyway while the true value should be around 7 or so
        # If you actually use that then you may not get very helpful variance reduction
        # like maybe at the beginning you get var red, but towards end you actually get variance increase
        # But by using the G - discounted_vals discounted all the way back to the start
        # what happens is much less importance is placed on the later time steps
        # So our value estimates are going to be wrong on the later time steps
        # But it doesn't really matter because since those G_ts are heavily discounted anyways
        # even a fairly large % mistake in the value won't have much influence on the true objective
        # so this should actually be significantly more stable
        # Has anyone analyzed this already?
        # Key characteristics: fixed termination/time steps, but that doesn't show up in the state
        # large enough discounting/long enough time horizon
        # I suppose this is kind of a hack that lets us tease apart/better take advantage of the fact that time steps play a big role.
        # If you had a time step variable in the state as well, then you shouldn't use this formulation, should use other formulation
        # An overestimate for the baseline value probably also helps in the same way that the adjustment_to_make_rewards_negative does

        # Hm ok let's try the other (correct) formulation and compare in the 2p case.

        # Yeah so interesting, because we cannot distinguish from 0,0 in time step 1 or 40, then maybe the baseline doens't help.
        # Or maybe it doesn't help much.

        # print(values_loss.shape)

        # regular_nl_loss = -(torch.cumsum(log_p_act, dim=0) * gamma_t_r_ts).sum(dim=0)

        # return dice_loss, G_ts, regular_nl_loss
        # return dice_loss, G_ts, values_loss

        return dice_loss, G_ts, values_loss, -dice_rewards, -dice_rewards2






# Of course these updates assume we have access to the reward model.

def contrib_game_with_func_approx(n, gamma=0.96, contribution_factor=1.6,
                                  contribution_scale=False):
    # Contribution game
    dims = [n] * n  # now each agent gets a vector observation, a n dimensional vector where each element is the action
    # of an agent, either 0 (defect) or 1 (coop) or 2 at the start of the game
    # print(dims)

    # Each player can choose to contribute 0 or contribute 1.
    # If you contribute 0 you get reward 1 as a baseline (or avoid losing 1)
    # The contribution is multiplied by the contribution factor c and redistributed
    # evenly among all agents, including those who didn't contribute
    # In the 2 player case, c needs to be > 1.5 otherwise CC isn't better than DD
    # And c needs to be < 2 otherwise C is not dominated by D for a selfish individual
    # But as the number of agents scales, we may consider scaling the contribution factor
    # contribution_factor = 1.7

    if contribution_scale:
        contribution_factor = contribution_factor * n
    else:
        assert contribution_factor > 1

    def Ls(th, num_iters=rollout_len):

        # stochastic rollout instead of matrix inversion (exact case)
        # helps significantly because even though it only saves ~n^3 time
        # the base n is already exponential in the number of agents as state space blows up exponentially
        # Rollouts can allow us to get 10, even 15 agents
        # But for 20 or 30 we will need func approx as well

        init_state = torch.Tensor([[init_state_representation] * n])  # repeat -1 (init state representation) n times, where n is num agents
        # Every agent sees same state; P1 [action, P2 action, P3 action ...]

        state = init_state

        trajectory = torch.zeros((num_iters, n_agents), dtype=torch.int)
        rewards = torch.zeros((num_iters, n_agents))
        policy_history = torch.zeros((num_iters, n_agents))

        discounts = torch.cumprod(gamma * torch.ones((num_iters)),dim=0) / gamma


        # TODO: rollout can be refactored as a function
        # Lots of this code can be refactored as functions


        for iter in range(num_iters):

            policies = torch.zeros(n_agents)

            for i in range(n_agents):
                if isinstance(th[i], torch.Tensor):

                    if (state - init_state).sum() == 0:
                        policy = torch.sigmoid(th[i])[-1]
                    else:

                        policy = torch.sigmoid(th[i])[game.int_from_bin_inttensor(state)]
                else:
                    policy = th[i](state)
                # policy prob of coop between 0 and 1
                policies[i] = policy

            policy_history[iter] = policies

            actions = torch.distributions.binomial.Binomial(probs=policies.detach()).sample()

            state = torch.Tensor(actions)

            trajectory[iter] = torch.Tensor(actions)

            # Note negative rewards might help with exploration in PG formulation
            total_contrib = sum(actions)
            payout_per_agent = total_contrib * contribution_factor / n
            agent_rewards = -actions + payout_per_agent  # if agent contributed 1, subtract 1, that's what the -actions does
            agent_rewards -= adjustment_to_make_rewards_negative
            rewards[iter] = agent_rewards

        G_ts = reverse_cumsum(rewards * discounts.reshape(-1,1), dim=0)

        # Discounted rewards, not sum of rewards which G_t is
        gamma_t_r_ts = rewards * discounts.reshape(-1,1)  # implicit broadcasting done by numpy

        p_act_given_state = trajectory.float() * policy_history + (
                    1 - trajectory.float()) * (
                                        1 - policy_history)  # recall 1 is coop, so when coop action 1 taken, we look at policy which is prob coop
        # and when defect 0 is taken, we take 1-policy = prob of defect

        log_p_act = torch.log(p_act_given_state)

        # These are basically grad_i E[R_0^i] - naive learning loss
        # no LOLA loss here yet
        losses_nl = (log_p_act * G_ts).sum(dim=0)
        # G_ts gives you the inner sum of discounted rewards

        log_p_times_G_t_matrix = torch.zeros((n_agents, n_agents))
        # so entry 0,0 is - (log_p_act[:,0] * G_ts[:,0]).sum(dim=0)
        # entry 1,1 is - (log_p_act[:,1] * G_ts[:,1]).sum(dim=0)
        # and so on
        # entry 0,1 is - (log_p_act[:,0] * G_ts[:,1]).sum(dim=0)
        # and so on
        # Be careful with dimensions/not to mix them up


        for i in range(n_agents):
            for j in range(n_agents):

                log_p_times_G_t_matrix[i][j] = (
                            log_p_act[:, i] * G_ts[:, j]).sum(dim=0)
        # Remember that the grad corresponds to the log_p and the R_t corresponds to the G_t
        # We can switch the log_p and G_t (swap log_p i to j and vice versa) if we want to change order

        # For the first term, my own derivation showed that
        # grad_2 E(R_1) = (prop to) Sum (grad_2 (log pi_2)) G_t(1)

        # For the grad_1 grad_2 term:
        # the way this will work is that the ith entry (row) in this log_p_act_sums_0_to_t
        # will be the sum of log probs from time 0 to time i
        # Then the dimension of each row is the number of agents - we have the sum of log probs
        # for each agent
        # later we will product them (but in pairwise combinations!)

        log_p_act_sums_0_to_t = torch.cumsum(log_p_act, dim=0)

        # Remember also that for p1 you want grad_1 grad_2 of R_2 (P2's return)
        # So then you also want grad_1 grad_3 of R_3
        # and so on

        grad_1_grad_2_matrix = torch.zeros((n_agents, n_agents))
        for i in range(n_agents):
            for j in range(n_agents):
                grad_1_grad_2_matrix[i][j] = (torch.FloatTensor(gamma_t_r_ts)[:,
                                              j] * log_p_act_sums_0_to_t[:,
                                                   i] * log_p_act_sums_0_to_t[:,
                                                        j]).sum(dim=0)
        # Here entry i j is grad_i grad_j E[R_j]

        losses = losses_nl

        grad_log_p_act = []
        for i in range(n_agents):
            # TODO could probably get this without taking grad, could be more efficient
            example_grad = get_gradient(log_p_act[0, i], th[i]) if isinstance(
                th[i], torch.Tensor) else torch.cat(
                [get_gradient(log_p_act[0, i], param).flatten() for
                 param in
                 th[i].parameters()])
            grad_len = len(example_grad)
            grad_log_p_act.append(torch.zeros((rollout_len, grad_len)))

        for i in range(n_agents):

            for t in range(rollout_len):

                grad_t = get_gradient(log_p_act[t, i], th[i]) if isinstance(
                    th[i], torch.Tensor) else torch.cat(
                    [get_gradient(log_p_act[t, i], param).flatten() for
                     param in
                     th[i].parameters()])


                grad_log_p_act[i][t] = grad_t

        return losses, grad_1_grad_2_matrix, log_p_times_G_t_matrix, G_ts, gamma_t_r_ts, log_p_act_sums_0_to_t, log_p_act, grad_log_p_act

        # TODO
        # Later use actor critic and other architectures

    return dims, Ls


def ipd(gamma=0.96):
    dims = [5, 5]
    payout_mat_1 = torch.Tensor([[-2, -5], [0, -4]])

    # payout_mat_1 = torch.Tensor([[-1, -3], [0, -2]])
    payout_mat_2 = torch.t(payout_mat_1)

    def Ls(th):
        # Theta denotes (unnormalized) action probabilities at each of the states:
        # start CC CD DC DD

        # Action prob at start of the game (higher val = more likely to coop)
        p_1_0 = torch.sigmoid(th[0][0:1])
        p_2_0 = torch.sigmoid(th[1][0:1])

        # Prob of each of the first states (after first action), CC CD DC DD
        p = torch.cat([p_1_0 * p_2_0,
                       p_1_0 * (1 - p_2_0),
                       (1 - p_1_0) * p_2_0,
                       (1 - p_1_0) * (1 - p_2_0)])
        # print(p)

        # Probabilities in the states other than the start state
        p_1 = torch.reshape(torch.sigmoid(th[0][1:5]), (4, 1))
        p_2 = torch.reshape(torch.sigmoid(th[1][1:5]), (4, 1))

        # Concat the col vectors above into a matrix
        # After multiplication, here we are creating the state transition matrix
        # So we first have p_1 * p_2, coop prob * coop prob, in each of the 4 states
        # Then we have CD prob in each of the 4 states, then DC prob, then DD prob
        # So it is like:
        #    CC CD DC DD
        # CC x  x  x  x
        # CD y  x  x  x
        # DC x  x  x  x
        # DD x  x  x  x
        # Where the row is the start state, and the column is the next state. So entry
        # y above denotes transition prob from CD to CC
        P = torch.cat([p_1 * p_2,
                       p_1 * (1 - p_2),
                       (1 - p_1) * p_2,
                       (1 - p_1) * (1 - p_2)], dim=1)

        # Now this is the matrix inversion for calculating the value function
        # (solving the bellman equation)
        # and of course first have p for the first step
        # The M is the (negative) discounted number of state visitations (?)
        # so it's like the IPD played in perpetuity but with discounting
        M = torch.matmul(p, torch.inverse(torch.eye(4) - gamma * P))

        # Then dot product with the payout for each player to return the 'loss'
        # which is actually just a negative reward
        # but over the course of the entire game naturally
        L_1 = torch.matmul(M, torch.reshape(payout_mat_1, (4, 1)))
        L_2 = torch.matmul(M, torch.reshape(payout_mat_2, (4, 1)))
        return [L_1, L_2]

    return dims, Ls


# Gradient computations for each algorithm.

def init_th(dims, std):
    th = []
    # Dims [5,5] or something, len is num agents
    # And each num represents the dim of the policy for that agent (equal to state space size with binary action/bernoulli dist)
    for i in range(len(dims)):
        if std > 0:
            init = torch.nn.init.normal_(
                torch.empty(dims[i], requires_grad=True), std=std)
        else:
            init = torch.zeros(dims[i], requires_grad=True)
        th.append(init)

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
    # Dims [5,5] or something, len is num agents
    # And each num represents the dim of the policy for that agent (equal to state space size with binary action/bernoulli dist)

    return th


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, extra_hidden_layers,
                 output_size):
        super(NeuralNet, self).__init__()
        layers = []

        # layers.append(torch.nn.Linear(input_size, output_size))

        layers.append(torch.nn.Linear(input_size, hidden_size))
        # layers.append(torch.nn.LeakyReLU(negative_slope=0.01))
        layers.append(torch.nn.Tanh())
        for i in range(extra_hidden_layers):
            layers.append(nn.Linear(hidden_size, hidden_size))
            # layers.append(torch.nn.LeakyReLU(negative_slope=0.01))
            layers.append(torch.nn.Tanh())
        layers.append(nn.Linear(hidden_size, output_size))
        layers.append(nn.Sigmoid())

        self.net = nn.Sequential(*layers)


    def forward(self, x):
        output = self.net(x)
        return output



def init_custom(dims):
    th = []


    # th.append(torch.nn.init.normal_(
    #     torch.empty(2 ** n_agents + 1, requires_grad=True), std=0.1))
    # th.append(
    #     NeuralNet(input_size=dims[0], hidden_size=16, extra_hidden_layers=0,
    #               output_size=1))

    # NN/func approx
    #
    # for i in range(len(dims)):
    #     th.append(
    #         NeuralNet(input_size=dims[i], hidden_size=16, extra_hidden_layers=0,
    #                   output_size=1))


    # Tabular policies

    for i in range(len(dims)):
        # DONT FORGET THIS +1
        # Right now if you omit the +1 we get a bug where the first state is the prob in the all contrib state
        th.append(torch.nn.init.normal_(torch.empty(2**n_agents + 1, requires_grad=True), std=0.1))


    # TFT init
    # logit_shift = 2
    # init = torch.zeros(5, requires_grad=True) - logit_shift
    # init[-1] += 2 * logit_shift
    # init[-2] += 2 * logit_shift
    # th.append(init)

    # optims = construct_optims_for_th(th, lrs=lr_policies)
    optims = None

    assert len(th) == len(dims)
    # assert len(optims) == len(dims)

    vals = []

    for i in range(len(dims)):
        vals.append(torch.nn.init.normal_(torch.empty(2**n_agents + 1, requires_grad=True), std=0.1))

    assert len(vals) == len(dims)

    return th, optims, vals

# def construct_optims_for_th(th, lrs):
#     optims = []
#     for i in range(len(th)):
#         if isinstance(th[i], NeuralNet):
#             optim = torch.optim.SGD(th[i].parameters(), lr=lrs[i])
#             diffoptim = higher.optim.DifferentiableSGD(optim, th[i].parameters())
#             optims.append(diffoptim)
#
#         else:
#             optim = torch.optim.SGD([th[i]], lr=lrs[i])
#             diffoptim = higher.optim.DifferentiableSGD(optim, [th[i]])
#             optims.append(diffoptim)
#     return optims


def get_gradient(function, param):
    grad = torch.autograd.grad(function, param, create_graph=True)[0]
    return grad


def print_policy_info(policy, i, G_ts, discounted_sum_of_adjustments, truncated_coop_payout, inf_coop_payout):

    print("Policy {}".format(i))
    print(
        "(Probabilities are for cooperation/contribution, for states 00...0 (no contrib,..., no contrib), 00...01 (only last player contrib), 00...010, 00...011, increasing in binary order ..., 11...11 , start)")

    print(policy)

    if i == 0:

        print(
            "Discounted Sum Rewards (Avg over batches) in this episode (removing negative adjustment): ")
        # print(G_ts[0] + discounted_sum_of_adjustments)

        print(G_ts[0].mean(dim=1).reshape(-1) + discounted_sum_of_adjustments)
        print("Max Avg Coop Payout (Truncated Horizon): {:.3f}".format(truncated_coop_payout))
        print("Max Avg Coop Payout (Infinite Horizon): {:.3f}".format(inf_coop_payout))


def print_value_info(vals, i):
    values = vals[i]
    print("Values {}".format(i))
    print(values)

def print_policies_from_state_batch(n_agents, G_ts, discounted_sum_of_adjustments, truncated_coop_payout, inf_coop_payout):
    state_batch = torch.cat((build_bin_matrix(n_agents, 2 ** n_agents),
                             torch.Tensor([
                                              init_state_representation] * n_agents).reshape(
                                 1, -1)))
    # policies = []
    for i in range(n_agents):
        if isinstance(th[i], torch.Tensor):
            policy = torch.sigmoid(th[i])
            # indices = list(map(int_from_bin_inttensor, state_batch))
            # indices[-1] = -1
            # print(indices)
            # print(policy[indices])
            # 1/0

        else:
            policy = th[i](state_batch)

        print_policy_info(policy, i, G_ts,
                          discounted_sum_of_adjustments, truncated_coop_payout, inf_coop_payout)

        # policies.append(policy)


def update_th(th, gradient_terms_or_Ls, lr_policies, eta, algos, epoch, using_samples):
    n = len(th)

    G_ts = None
    grad_2_return_1 = None
    nl_terms = None

    if using_samples:
        # losses, grad_1_grad_2_matrix, log_p_times_G_t_matrix, G_ts, gamma_t_r_ts, log_p_act_sums_0_to_t, log_p_act, grad_log_p_act = Ls(
        #     th)
        losses, grad_1_grad_2_matrix, log_p_times_G_t_matrix, G_ts, gamma_t_r_ts, log_p_act_sums_0_to_t, log_p_act, grad_log_p_act = gradient_terms_or_Ls

    else:
        losses = gradient_terms_or_Ls(th)

    # Compute gradients
    # This is a 2d array of all the pairwise gradient computations between all agents
    # This is useful for LOLA and the other opponent modeling stuff
    # So it is the gradient of the loss of agent j with respect to the parameters of agent i
    # When j=i this is just regular gradient update


    if not using_samples:

        grad_L = [[get_gradient(losses[j], th[i]) for j in range(n)] for i in
                  range(n)]

    # calculate grad_L as the gradient of loss for player j with respect to parameters for player i
    # Therefore grad_L[i][i] is simply the naive learning loss for player i

    # if algo == 'lola':
    if 'lola' in algos:


        if using_samples:

            grad_1_grad_2_return_2_new = []
            for i in range(n_agents):
                grad_1_grad_2_return_2_new.append([0] * n_agents)


            grad_log_p_act_sums_0_to_t = []
            for i in range(n_agents):
                grad_log_p_act_sums_0_to_t.append(torch.cumsum(grad_log_p_act[i], dim=0))
            # print(grad_log_p_act_sums_0_to_t)

            for i in range(n_agents):
                for j in range(n_agents):
                    if i != j:
                        for t in range(rollout_len):
                            for b in range(batch_size):

                                # grad_t = torch.FloatTensor(gamma_t_r_ts)[:, j][t] * \
                                #          torch.outer(
                                #              grad_log_p_act[i][:t + 1].sum(dim=0),
                                #              grad_log_p_act[j][:t + 1].sum(dim=0))


                                # print(grad_log_p_act_sums_0_to_t[i].shape)
                                # print(torch.FloatTensor(gamma_t_r_ts)[:, j][t].shape)
                                # print(grad_log_p_act_sums_0_to_t[i][t].shape)
                                # print(grad_log_p_act_sums_0_to_t[j][t].shape)


                                grad_t_b = torch.FloatTensor(gamma_t_r_ts)[:, j][t][b] * \
                                         torch.outer(
                                             grad_log_p_act_sums_0_to_t[i][t][b],
                                             grad_log_p_act_sums_0_to_t[j][t][b])

                                if t == 0 and b == 0:
                                    grad_1_grad_2_return_2_new[i][j] = grad_t_b
                                else:
                                    grad_1_grad_2_return_2_new[i][j] += grad_t_b
                            grad_1_grad_2_return_2_new[i][j] /= batch_size # do an average


            grad_1_grad_2_return_2 = grad_1_grad_2_return_2_new


            grad_2_return_1 = [
                [get_gradient(log_p_times_G_t_matrix[j][i],
                              th[j]) if isinstance(th[j], torch.Tensor) else
                 torch.cat([get_gradient(log_p_times_G_t_matrix[j][i],
                                         param).flatten() for param in
                            th[j].parameters()])
                 for j in range(n)]
                for i in range(n)]

            # print(grad_L)
            # print(grad_2_return_1)

            # By the way these matrix names I came up with suck because they don't actually have the grad
            # We will actually take the grad here
            # By taking grad_1_grad_2_return_2[j][j] what we do is we have the E[R_2] term (second j), which we then take the grad
            # with respect to p2 (first j) (term B)
            # Then for grad_2_return_1[j][i], we take the E[R_1] term (second position, i)
            # and then take the grad with respect to p2 (first position, j) (term A)
            # Finally we will take the grad with respect to p1 of A * B
            # And since A has no term with respect to p1 (yes we are taking expectation over p1 action, but expectation
            # is approximated by sample, so after that, we have no p1 term showing up anywhere anymore)
            # Then this will work and we will get what we want.

            # Eta is a hyperparam

            # IMPORTANT NOTE: the way these grad return matrices are set up is that you should always call i j here
            # because the j i switch was done during the construction of the matrix
            # TODO allow for different eta across agents?
            # And then account for that here.
            lola_terms = [sum([eta * grad_2_return_1[i][j].t() @
                               grad_1_grad_2_return_2[i][j].t() for j in
                               range(n) if j != i]) for i in range(n)]

            grads = []
            for i in range(n):
                if isinstance(th[i], torch.Tensor):
                    grads.append(
                        (grad_2_return_1[i][i] + lola_terms[i]) if algos[
                                                                       i] == 'lola' else
                        grad_2_return_1[i][i])
                else:
                    if algos[i] == 'lola':
                        grad = []
                        start_pos = 0

                        for (k, param) in enumerate(th[i].parameters()):


                            param_len = len(param.flatten())

                            grad.append(
                                (grad_2_return_1[i][i][start_pos:start_pos + param_len] + lola_terms[i][
                                                           start_pos:start_pos + param_len]).reshape(param.size())
                            )
                            start_pos += param_len


                        grads.append(grad)
                    else:
                        grads.append(grad_2_return_1[i][i])

        else:
            # Look at pg 12, Stable Opponent Shaping paper
            # This is literally the first line of the second set of equations
            # sum over all j != i of grad_j L_i * grad_j L_j
            # Then that's your lola term once you differentiate it with respect to theta_i
            # Then add the naive learning term
            # But this is the SOS formulation, not the LOLA one. SOS differentiates through
            # more than the LOLA one does.
            terms = [sum([torch.dot(grad_L[j][i], grad_L[j][j])
                          for j in range(n) if j != i]) for i in range(n)]

            lola_terms = [
                # lr_policies[i] *
                eta * get_gradient(terms[i], th[i])
                for i in range(n)]

            nl_terms = [grad_L[i][i]
                        for i in range(n)]

            grads = [nl_terms[i] + lola_terms[i] for i in range(n)]

    else:  # Naive Learning
        grads = [grad_L[i][i] for i in range(n)]

    # Update theta
    with torch.no_grad():
        for i in range(n):
            # if using_samples:
            if not isinstance(th[i], torch.Tensor):
                k = 0
                for param in th[i].parameters():
                    param += lr_policies[i] * grads[i][k]
                    k += 1
            else:

                th[i] += lr_policies[i] * grads[i]
    return th, losses, G_ts, nl_terms, lola_terms, grad_2_return_1




# Main loop/code
if __name__ == "__main__":
    parser = argparse.ArgumentParser("NPLOLA")
    parser.add_argument("--using_samples", action="store_true",
                        help="True for samples (with rollout_len), false for exact gradient (using matrix inverse for infinite length rollout)")
    parser.add_argument("--using_DiCE", action="store_true",
                        help="True for LOLA-DiCE, false for LOLA-PG. Must have using_samples = True.")
    parser.add_argument("--repeat_train_on_same_samples", action="store_true",
                        help="True for PPO style formulation where we repeat train on the same samples (only one inner step rollout, multiple inner step updates with importance weighting)")
    parser.add_argument("--use_clipping", action="store_true",
                        help="Do the PPO style clipping")
    parser.add_argument("--clip_epsilon", type=float, default=0.2, help="PPO style clip hyperparameter")
    parser.add_argument("--gamma", type=float, default=0.96, help="discount rate")
    parser.add_argument("--print_every", type=int, default=200, help="Print every x number of epochs")
    parser.add_argument("--num_epochs", type=int, default=50001, help="number of epochs to run")
    parser.add_argument("--repeats", type=int, default=1, help="repeats per setting configuration")
    parser.add_argument("--n_agents_list", nargs="+", type=int, default=[5],
                        help="list of number of agents to try")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--etas", nargs="+", type=int, default=[20, 12],
                        help="list of etas to try")
    parser.add_argument("--lr_policies", type=float, default=0.05,
                        help="same learning rate across all policies for now")
    parser.add_argument("--lr_values_scale", type=float, default=0.5,
                        help="scale lr_values relative to lr_policies")

    args = parser.parse_args()

    # Repeats for each hyperparam setting
    # repeats = 10
    repeats = args.repeats

    # tanh instead of relu or lrelu activation seems to help. Perhaps the gradient flow is a bit nicer that way

    # For each repeat/run:
    num_epochs = args.num_epochs
    # print_every = max(1, num_epochs / 50)
    # print_every = 200
    print_every = args.print_every
    batch_size = args.batch_size
    # Bigger batch is a big part of convergence with DiCE. Too small batch (e.g. 1 or 4) frequently results in issues.

    gamma = args.gamma

    using_samples = args.using_samples
    using_DiCE = args.using_DiCE
    if using_DiCE:
        assert using_samples
        repeat_train_on_same_samples = args.repeat_train_on_same_samples  # If true we will instead of rolling out multiple times in the inner loop, just rollout once
        # but then train multiple times on the same data using importance sampling and PPO-style clipping
        use_clipping = args.use_clipping
        clip_epsilon = args.clip_epsilon
    # TODO it seems the non-DiCE version with batches isn't really working.

    # For DiCE
    symmetric_updates = False  # Not done for now

    # # Why does LOLA agent sometimes defect at start but otherwise play TFT? Policy gradient issue?
    # etas = [
    #     0.01 * 5]  # wait actually this doesn't seem to work well at all... no consistency in results without dice... is it because we missing 1 term? this is batch size 1
    # if not using_samples:
    #     etas = [0.05 * 20, 0.05 * 12]



    n_agents_list = args.n_agents_list
    # n_agents_list = [5, 8]


    for n_agents in n_agents_list:

        assert n_agents >= 2
        if n_agents == 2:
            contribution_factor = 1.6
            contribution_scale = False
        else:
            contribution_factor = 0.6
            contribution_scale = True

        lr_policies = torch.tensor([args.lr_policies] * n_agents)

        lr_values = lr_policies * args.lr_values_scale
        # lr_values = lr_policies * 0.2

        # etas = [0.05 * 20, 0.05 * 12]  # for both exact and pg formulations
        etas = [eta * args.lr_policies for eta in args.etas]
        # etas = args.etas * args.lr_policies

        if using_DiCE:
            etas = args.etas  # [8] # [20] # this is a factor by which we increase the lr on the inner loop vs outer loop

        if not contribution_scale:
            # inf = infinite horizon
            inf_coop_payout = 1 / (1 - gamma) * (contribution_factor - 1)
            truncated_coop_payout = inf_coop_payout * \
                          (
                                      1 - gamma ** rollout_len)  # This last term here accounts for the fact that we don't go to infinity
            inf_max_payout = 1 / (1 - gamma) * (
                        contribution_factor * (n_agents - 1) / n_agents)
            truncated_max_payout = inf_max_payout * \
                         (1 - gamma ** rollout_len)
        else:
            inf_coop_payout = 1 / (1 - gamma) * (contribution_factor * n_agents - 1)
            truncated_coop_payout = inf_coop_payout * \
                          (1 - gamma ** rollout_len)
            inf_max_payout = 1 / (1 - gamma) * (contribution_factor * (n_agents - 1))
            truncated_max_payout = inf_max_payout * \
                         (1 - gamma ** rollout_len)

        max_single_step_return = (contribution_factor * (n_agents - 1) / n_agents)

        adjustment_to_make_rewards_negative = 0
        # adjustment_to_make_rewards_negative = max_single_step_return
        # With adjustment and 20k steps seems LOLA vs NL does learn a TFT like strategy
        # But the problem is NL hasn't learned to coop at the start
        # which results in DD behaviour throughout.

        discounted_sum_of_adjustments = 1 / (
                    1 - gamma) * adjustment_to_make_rewards_negative * \
                                        (1 - gamma ** rollout_len)


        for eta in etas:

            print("Number of agents: {}".format(n_agents))
            print("Contribution factor: {}".format(contribution_factor))
            print("Scaled contribution factor? {}".format(contribution_scale))
            print("Eta: {}".format(eta))
            # print(reward_percent_of_max)
            # Average over all runs
            if not using_samples:
                print("Exact Gradients")
            else:
                if using_DiCE:
                    if symmetric_updates:
                        print("Symmetric DiCE Updates")
                    else:
                        print("Asymmetric DiCE Updates")
                    if repeat_train_on_same_samples:
                        print("Using Repeat Train on Same Samples")
                    else:
                        print("Using regular DiCE formulation")
                else:
                    print("Policy Gradient Updates")

            reward_percent_of_max = []

            for run in range(repeats):

                if not using_samples:

                    dims, Ls = ipdn(n=n_agents, gamma=gamma,
                                    contribution_factor=contribution_factor,
                                    contribution_scale=contribution_scale)


                    std = 1
                    if theta_init_mode == 'tft':
                        # std = 0.1
                        # Basically with std higher, you're going to need higher logit shift (but only slightly, really), in order to reduce the variance
                        # and avoid random-like behaviour which could undermine the closeness/pull into the TFT basin of solutions
                        th = init_th_tft(dims, std, logit_shift=1.7)
                        # Need around 1.85 for NL and 1.7 for LOLA
                    else:
                        th = init_th(dims, std)


                else:
                    # Using samples instead of exact here


                    # dims, Ls = contrib_game_with_func_approx(n=n_agents, gamma=gamma,
                    #                                          contribution_factor=contribution_factor,
                    #                                          contribution_scale=contribution_scale)

                    game = ContributionGame(n=n_agents, gamma=gamma, batch_size=batch_size, num_iters=rollout_len,
                                                             contribution_factor=contribution_factor,
                                                             contribution_scale=contribution_scale)
                    dims = game.dims

                    th, _, vals = init_custom(dims)

                    # I think part of the issue is if policy saturates at cooperation it never explores and never tries defect
                    # How does standard reinforce/policy gradient get around this? Entropy regularization
                    # Baseline might help too. As might negative rewards everywhere.


                if using_DiCE:
                    # inner_steps = [2,2]
                    inner_steps = [2] * n_agents
                else:
                    # algos = ['nl', 'lola']
                    # algos = ['lola', 'nl']
                    algos = ['lola'] * n_agents


                # Run
                if using_samples:
                    G_ts_record = torch.zeros((num_epochs, n_agents, batch_size, 1))
                else:
                    G_ts_record = torch.zeros(
                        (num_epochs, n_agents))
                lola_terms_running_total = []
                nl_terms_running_total = []


                # th_out = []

                accum_diffs = [None]*n_agents
                for i in range(n_agents):
                    accum_diffs[i] = torch.zeros(5)
                for epoch in range(num_epochs):
                    if using_samples:
                        if using_DiCE:
                            static_th_copy = copy.deepcopy(th)
                            static_vals_copy = copy.deepcopy(vals)

                            # optims_primes = construct_optims_for_th(theta_primes)

                            # TODO Try actual symmetric with inner steps now

                            if symmetric_updates:
                                raise NotImplementedError
                                # assert inner_steps[0] == sum(inner_steps) / len(inner_steps)
                                # theta_primes = copy.deepcopy(static_th_copy)
                                #
                                # # Allow for inner steps
                                # if eta != 0:
                                #     K = inner_steps[0]
                                #     assert K == 1
                                #     # With 1 inner step right now
                                #     # There's a non-trivial question how you would allow for multiple steps
                                #     # IMPORTANT NOTE THAT APPLIES FOR ASYMM TOO
                                #     # With multiple players and steps > 1 the differentiable update step
                                #     # means that other players also do a kind of LOLA update in the inner loop
                                #     # This may or may not be what you want...
                                #
                                #     for step in range(inner_steps[0]):
                                #
                                #         grads = [None] * n_agents
                                #
                                #
                                #         trajectory, rewards, policy_history = game.rollout(
                                #             th)
                                #
                                #         dice_loss, _ = game.get_dice_loss(
                                #             trajectory, rewards,
                                #             policy_history)
                                #         for j in range(n_agents):
                                #             # print(th[j])
                                #             if isinstance(th[j], torch.Tensor):
                                #                 grad = get_gradient(dice_loss[j], th[j])
                                #                 grads[j] = grad
                                #             else:
                                #                 1/0
                                #
                                #         for j in range(n_agents):
                                #             # print(th[j])
                                #             if isinstance(theta_primes[j], torch.Tensor):
                                #                 theta_primes[j] = theta_primes[j] - \
                                #                                   lr_policies[
                                #                                       j] * eta * grads[j]
                                #             else:
                                #                 1 / 0
                                #
                                # for i in range(n_agents):
                                #     mixed_thetas = theta_primes
                                #     mixed_thetas[i] = th[i]
                                #     trajectory, rewards, policy_history = game.rollout(
                                #         mixed_thetas)
                                #
                                #     dice_loss, G_ts = game.get_dice_loss(
                                #         trajectory, rewards,
                                #         policy_history)
                                #
                                #     if isinstance(th[i], torch.Tensor):
                                #         grad = get_gradient(dice_loss[i], th[i])
                                #         with torch.no_grad():
                                #             th[i] -= lr_policies[i] * grad
                                #
                                #     else:
                                #         optim_update(optims[i], dice_loss[i])




                            else:

                                # if repeat_train_on_same_samples:
                                #     mixed_thetas = copy.deepcopy(static_th_copy)
                                #     mixed_vals = copy.deepcopy(static_vals_copy)
                                #     trajectory, rewards, policy_history, val_history = game.rollout(
                                #         mixed_thetas, mixed_vals)
                                # challenge is that you need to reset these mixed_thetas for each agent
                                # how can you do that without rerolling? loss backward and all this stuff too


                                for i in range(n_agents):
                                    K = inner_steps[i]

                                    # TODO later confirm that this deepcopy is working properly for NN also
                                    theta_primes = copy.deepcopy(static_th_copy)
                                    val_primes = copy.deepcopy(static_vals_copy)
                                    # optims_primes = construct_optims_for_th(theta_primes, lrs=lr_policies * eta)

                                    # if not repeat_train_on_same_samples:
                                    mixed_thetas = theta_primes
                                    mixed_thetas[i] = th[i]
                                    mixed_vals = val_primes
                                    mixed_vals[i] = vals[i]

                                    # print(mixed_thetas)
                                    if eta != 0:

                                        if repeat_train_on_same_samples:
                                            trajectory, rewards, policy_history, val_history, next_val_history = game.rollout(
                                                mixed_thetas, mixed_vals)
                                            for step in range(K):
                                                if step == 0:
                                                    # dice_loss, _, values_loss = game.get_dice_loss(
                                                    #     trajectory,
                                                    #     rewards,
                                                    #     policy_history, val_history)
                                                    dice_loss, _, values_loss, r1,r2 = game.get_dice_loss(
                                                        trajectory,
                                                        rewards,
                                                        policy_history, val_history, next_val_history, old_policy_history=policy_history)
                                                else:
                                                    new_policies, new_vals, next_new_vals = game.get_policies_vals_for_states(
                                                        mixed_thetas, mixed_vals, trajectory)
                                                    # Using the new policies and vals now
                                                    # Always be careful not to overwrite/reuse names of existing variables
                                                    dice_loss, _, values_loss,r1,r2 = game.get_dice_loss(
                                                        trajectory, rewards, new_policies, new_vals, next_new_vals, old_policy_history=policy_history)

                                                grads = [None] * n_agents
                                                for j in range(n_agents):
                                                    if j != i:

                                                        if isinstance(mixed_thetas[j],
                                                                      torch.Tensor):
                                                            grad = get_gradient(
                                                                dice_loss[j],
                                                                mixed_thetas[j])
                                                            grads[j] = grad

                                                        else:
                                                            1/0
                                                for j in range(n_agents):
                                                    # Need a separate loop to deal with issue where you can't update differentiably first in p2 otherwise p3 will see that and diff through that
                                                    if j != i:
                                                        if isinstance(mixed_thetas[j],
                                                                      torch.Tensor):

                                                            mixed_thetas[j] = \
                                                            mixed_thetas[j] - \
                                                            lr_policies[
                                                                j] * eta * grads[
                                                                j]  # This step is critical to allow the gradient to flow through
                                                        else:
                                                            1 / 0
                                                # TODO this is unclipped, try clipped afterwards
                                                # Also TODO Aug 23 is do an outer loop with number of steps also




                                        else:

                                            for step in range(K):

                                            # ok to use th[i] here because it hasn't been updated yet
                                                trajectory, rewards, policy_history, val_history, next_val_history = game.rollout(
                                                    mixed_thetas, mixed_vals)
                                                dice_loss, _, values_loss,r1,r2 = game.get_dice_loss(
                                                    trajectory,
                                                    rewards,
                                                    policy_history, val_history, next_val_history)



                                                grads = [None] * n_agents
                                                for j in range(n_agents):
                                                    if j != i:
                                                          # updated_vals = optim_update(optims_primes[j],
                                                          #              dice_loss[j], [mixed_thetas[j]])
                                                          # mixed_thetas[j] = updated_vals[0]


                                                          if isinstance(mixed_thetas[j], torch.Tensor):
                                                              grad = get_gradient(dice_loss[j], mixed_thetas[j])
                                                              grads[j] = grad

                                                              # a = get_gradient(
                                                              #     r1[j],
                                                              #     mixed_thetas[j])
                                                              # b = get_gradient(
                                                              #     r2[j],
                                                              #     mixed_thetas[j])
                                                              # c = get_gradient(
                                                              #     a[0],
                                                              #     mixed_thetas[i])
                                                              # d = get_gradient(
                                                              #     b[0],
                                                              #     mixed_thetas[i])
                                                              # # print(b-a)
                                                              # print(d-c)
                                                              # grads[j] += (b-a) # a + b-a = b
                                                              # grads[j] += (b-a).detach() # This works ok
                                                              # # grads[j] += a-b # b + a-b = a

                                                              # accum_diffs[j] += (
                                                              #             b-a)
                                                              # print(accum_diffs)


                                                          else:
                                                              # TODO also the update needs to go outside this loop
                                                              # Get all the grads first, then update all simultaneously
                                                              # Don't update first because then gradient will flow through to other players
                                                              # updates as well
                                                              # for param in mixed_thetas[j].parameters():
                                                              #     print(param)
                                                              #     param.data +=  1
                                                              #     print(param)
                                                              #     # grad = get_gradient(dice_loss[j],
                                                              #     #           param)
                                                              #     # print(grad)
                                                              # for param in mixed_thetas[j].parameters():
                                                              #     print(param)
                                                              #
                                                              # print(mixed_thetas[j].parameters())
                                                              # 1 / 0
                                                              # grad = [get_gradient(dice_loss[j],
                                                              #               param) for param in mixed_thetas[j].parameters()]
                                                              # for i in range(len(grad)):
                                                              #     pass
                                                              1/0
                                                              # TODO we have to use higher.
                                                              # TODO Figure out how to use it, and use it.
                                                              # Figure out the unroll loop...

                                                              # optim_update(optims_primes[j], dice_loss[j])
                                                for j in range(n_agents):
                                                    # Need a separate loop to deal with issue where you can't update differentiably first in p2 otherwise p3 will see that and diff through that
                                                    if j != i:
                                                        if isinstance(mixed_thetas[j],
                                                                      torch.Tensor):

                                                            mixed_thetas[j] = mixed_thetas[j] - \
                                                                              lr_policies[
                                                                                  j] * eta * grads[j]  # This step is critical to allow the gradient to flow through
                                                            # You cannot use torch.no_grad on this step
                                                            # with torch.no_grad():
                                                            #     mixed_thetas[j] += lr_policies[j] * grad
                                                        else:
                                                            1/0


                                                # optim_update(optims_primes[j],
                                                #              dice_loss[j])

                                                # print(mixed_thetas[j])
                                                # print(th[j])
                                                # print(theta_primes[j])
                                                # print(static_th_copy[j])

                                    # TODO: Batch with LOLA non DiCE and test it
                                    # TODO: try to get 3p DiCE working, debug every step thoroughly!!!!
                                    # ABOVE KEY
                                    # And if it is still not working, investigate why it doesn't, since
                                    # We know that we can get the other non-dice variant working
                                    # DO these before the diff opt and NN and whatever.


                                    # Now calculate outer step using for each player a mix of the theta_primes and old thetas

                                    # mixed_thetas = theta_primes
                                    # mixed_thetas[i] = th[i]
                                    # ok to use th[i] here because it hasn't been updated yet
                                    # (the th[i-1] may have been udpated but that's ok because we don't use that)

                                    # mixed_optims = optims_primes
                                    # mixed_optims[i] = optims[i]

                                    if repeat_train_on_same_samples:
                                        mixed_thetas[i] = th[i]
                                        mixed_vals[i] = vals[i]

                                    trajectory, rewards, policy_history, val_history, next_val_history = game.rollout(
                                        mixed_thetas, mixed_vals)


                                    if repeat_train_on_same_samples:
                                        dice_loss, G_ts, values_loss,_,_ = game.get_dice_loss(
                                            trajectory, rewards,
                                            policy_history, val_history, next_val_history,
                                            old_policy_history=policy_history)
                                    else:
                                        dice_loss, G_ts, values_loss,_,_ = game.get_dice_loss(
                                            trajectory, rewards,
                                            policy_history, val_history, next_val_history)


                                    # NOTE: TODO potentially: G_ts here may not be the best choice
                                    # But it should be close enough to give an idea of what the rewards roughly look like
                                    if isinstance(th[i], torch.Tensor):
                                        grad = get_gradient(dice_loss[i], th[i])
                                        grad_val = get_gradient(values_loss[i], vals[i])

                                        # a = get_gradient(r1[i], th[i])
                                        # b = get_gradient(r2[i], th[i])
                                        # print(a)
                                        # print(b)
                                        # print(b-a)
                                        # accum_diffs[i] += (b-a)
                                        # print(accum_diffs)

                                        with torch.no_grad():
                                            th[i] -= lr_policies[i] * grad
                                            vals[i] -= lr_values[i] * grad_val
                                            # th[i] -= lr_policies[i] * (b-a)

                                        # TODO Be careful with +/- formulation now...
                                        # TODO rename the non-DiCE terms as rewards
                                        # And keep the DiCE terms as losses


                                    else:
                                        # optim_update(optims[i], dice_loss[i])
                                        1/0

                                    # optim_update(optims[i], dice_loss[i])

                                    # print(mixed_thetas)
                                    # print(th)
                                    # print("---")



                        else:

                            trajectory, rewards, policy_history = game.rollout(th)
                            gradient_terms = game.get_gradient_terms(trajectory, rewards, policy_history)

                            th, losses, G_ts, nl_terms, lola_terms, grad_2_return_1 = \
                                update_th(th, gradient_terms, lr_policies, eta, algos, using_samples=using_samples, epoch=epoch)
                    else:
                        th, losses, G_ts, nl_terms, lola_terms, grad_2_return_1 = \
                            update_th(th, Ls, lr_policies, eta, algos, using_samples=using_samples, epoch=epoch)


                    if using_samples:
                        assert G_ts is not None
                        G_ts_record[epoch] = G_ts[0]
                    else:
                        G_ts_record[epoch] = torch.stack(losses).detach()

                    # This is just to get an idea of the relative influence of the lola vs nl terms
                    # if lola_terms_running_total == []:
                    #     for i in range(n_agents):
                    #         lola_terms_running_total.append(lr_policies[i] * lola_terms[i].detach())
                    # else:
                    #     for i in range(n_agents):
                    #         lola_terms_running_total[i] += lr_policies[i] * lola_terms[i].detach()
                    #
                    # for i in range(n_agents):
                    #     if grad_2_return_1 is None:
                    #         nl_term = nl_terms[i].detach()
                    #     else:
                    #         nl_term = grad_2_return_1[i][i].detach()
                    #
                    #     if len(nl_terms_running_total) < n_agents:
                    #         nl_terms_running_total.append(lr_policies[i] * nl_term)
                    #     else:
                    #         nl_terms_running_total[i] += lr_policies[i] * nl_term

                    if epoch % print_every == 0:
                        print("Epoch: " + str(epoch))
                        print("Eta: " + str(eta))
                        print("Batch size: " + str(batch_size))
                        if using_DiCE:
                            print("Inner Steps: {}".format(inner_steps))
                        else:
                            print("Algos: {}".format(algos))
                        print("lr_policies: {}".format(lr_policies))
                        print("lr_values: {}".format(lr_values))
                        # print("LOLA Terms: ")
                        # print(lola_terms_running_total)
                        # print("NL Terms: ")
                        # print(nl_terms_running_total)
                        if not using_samples:
                            print("Discounted Rewards (Objective): {}".format(losses))
                            print(
                                "Max Avg Coop Payout (Infinite Horizon): {:.3f}".format(
                                    inf_coop_payout))


                        # Print policies here
                        if using_samples:
                            print_policies_from_state_batch(n_agents, G_ts,
                                                        discounted_sum_of_adjustments,
                                                        truncated_coop_payout, inf_coop_payout)
                            for i in range(n_agents):
                                print_value_info(vals, i)
                        else:
                            for i in range(n_agents):
                                policy = torch.sigmoid(th[i])
                                print("Policy {}".format(i))
                                print(policy)


                # % comparison of average individual reward to max average individual reward
                # This gives us a rough idea of how close to optimal (how close to full cooperation) we are.
                reward_percent_of_max.append((G_ts_record.mean() + discounted_sum_of_adjustments) / truncated_coop_payout)

                # print(reward_percent_of_max)
                # print(G_ts_record)


                plot_results = True
                if plot_results:
                    now = datetime.datetime.now()
                    # print(now.strftime('%Y-%m-%d_%H-%M'))
                    if using_samples:
                        avg_gts_to_plot = (G_ts_record + discounted_sum_of_adjustments).mean(dim=2).view(num_epochs, n_agents)
                    else:
                        avg_gts_to_plot = G_ts_record
                    # print(avg_gts_to_plot)
                    plt.plot(avg_gts_to_plot)
                    if using_samples:
                        plt.savefig("{}agents_{}eta_run{}_steps{}_date{}.png".format(n_agents, eta, run, "_".join(list(map(str, inner_steps))), now.strftime('%Y-%m-%d_%H-%M')))
                    else:
                        plt.savefig("{}agents_{:.0f}eta_run{}_exact_date{}.png".format(n_agents, eta / args.lr_policies, run, now.strftime('%Y-%m-%d_%H-%M')))

                    # plt.show()


            print("Average reward as % of max: {:.1%}".format(
                sum(reward_percent_of_max) / repeats))

