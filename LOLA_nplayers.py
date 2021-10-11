import numpy as np
import torch
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F

import higher

import datetime

import copy

import argparse

import random



# init_state_representation = 2  # Change here if you want different number to represent the initial state
rollout_len = 50

theta_init_modes = ['standard', 'tft']
theta_init_mode = 'standard'






def bin_inttensor_from_int(x, n):
    """Converts decimal value integer x into binary representation.
    Parameter n represents the number of agents (so you fill with 0s up to the number of agents)
    Well n doesn't have to be num agents. In case of lookback (say 2 steps)
    then we may want n = 2x number of agents"""
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


def copyNN(copy_to_net, copy_from_net):
    # Copy from curr to target
    copy_to_net.load_state_dict(copy_from_net.state_dict())

def optim_update(optim, loss, params=None):
    if params is not None:
        #diffopt step here
        return optim.step(loss, params)
    else:
        optim.zero_grad()
        loss.backward(retain_graph=True)
        optim.step()

# def optim_update(optim, loss):
#     optim.zero_grad()
#     loss.backward(retain_graph=True)
#     optim.step()

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
    """
    Because the way this is structured, 1 means a contribution of 1 (and is therefore cooperation) and 0 is a contribution of 0, which is defecting
    The game works conceptually as: at each round, an agent can either contribute 0 or 1.
    The total number of contributions go into a public pool (e.g. consider some investment fund, or investing in infrastructure, or something along those lines)
    which is redistributed equally to agents (all agents benefit equally from the public investment/infrastructure).
    The value each agent gets from investing 1 must be <1 for the agent itself, but the total value (if you sum up the value each individual agent has across all agents)
    must be > 1. (So if contribution_scale = False, contribution_factor needs to be > 1, otherwise nobody should ever contribute)
    Contributing 1 provides an individual reward of -1 (in addition to the redistribution of total contributions)
    Contributing 0 provides no individual reward (beyond that from the redistribution of total contributions)
    """
    def __init__(self, n, batch_size, num_iters, gamma=0.96, contribution_factor=1.6, contribution_scale=False, history_len=1, using_mnist_states=False):

        self.n_agents = n
        self.gamma = gamma
        self.contribution_factor = contribution_factor
        self.contribution_scale = contribution_scale
        self.history_len = history_len
        self.dims = [n * history_len] * n
        self.batch_size = batch_size
        self.num_iters = num_iters
        self.using_mnist_states = using_mnist_states

        if contribution_scale:
            self.contribution_factor = contribution_factor * n
        else:
            assert self.contribution_factor > 1

        self.dec_value_mask = (2 ** torch.arange(n - 1, -1, -1)).float()


        if self.using_mnist_states:
            from torchvision import datasets, transforms

            mnist_train = datasets.MNIST('data', train=True, download=True,
                                         transform=transforms.ToTensor())
            # print(mnist_train[0][0])
            self.coop_class = 1
            self.defect_class = 0
            idx_coop = (mnist_train.targets) == self.coop_class
            idx_defect = (mnist_train.targets) == self.defect_class
            idx_init = (mnist_train.targets) == init_state_representation

            self.mnist_coop_class_dset = torch.utils.data.dataset.Subset(
                mnist_train,
                np.where(
                    idx_coop == 1)[
                    0])
            self.mnist_defect_class_dset = torch.utils.data.dataset.Subset(
                mnist_train,
                np.where(
                    idx_defect == 1)[
                    0])
            self.mnist_init_class_dset = torch.utils.data.dataset.Subset(
                mnist_train,
                np.where(
                    idx_init == 1)[
                    0])
            self.len_mnist_coop_dset = len(self.mnist_coop_class_dset)
            self.len_mnist_defect_dset = len(self.mnist_defect_class_dset)
            self.len_mnist_init_dset = len(self.mnist_init_class_dset)
            # print(torch.randint(0, self.len_mnist_init_dset, (self.batch_size,)))
            #
            # print(self.mnist_defect_class_dset[0][1])
            #
            # idx = torch.tensor(mnist_train.targets) == defect_class
            # idx = mnist_train.train_labels == 1
            # print(idx)
            # labels = mnist_train.train_labels[idx]
            # data = mnist_train.train_data[idx][0]
            # print(labels)
            # print(data)

    def build_mnist_state_from_classes(self, batch_tensor):
        dims = batch_tensor.shape

        mnist_state = torch.zeros((dims[0], dims[1], 28, 28))
        for b in range(dims[0]):
            for c in range(dims[1]):
                mnist_class = batch_tensor[b][c]
                if mnist_class == init_state_representation:
                    randind = random.randint(0, self.len_mnist_init_dset - 1)
                    mnist_img = self.mnist_init_class_dset[randind][0]
                elif mnist_class == self.coop_class:
                    randind = random.randint(0, self.len_mnist_coop_dset - 1)
                    mnist_img = self.mnist_coop_class_dset[randind][0]
                else:
                    assert mnist_class == self.defect_class
                    randind = random.randint(0, self.len_mnist_defect_dset - 1)
                    mnist_img = self.mnist_defect_class_dset[randind][0]
                mnist_state[b][c] = mnist_img

        return mnist_state

    def get_init_state_batch(self):
        if self.using_mnist_states:
            integer_state_batch = torch.ones(
                (self.batch_size,
                 self.n_agents * self.history_len)) * init_state_representation
            init_state_batch = self.build_mnist_state_from_classes(integer_state_batch)

        else:
            init_state_batch = torch.ones(
                (self.batch_size, self.n_agents * self.history_len)) * init_state_representation
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

    def get_policy_and_state_value(self, pol, val, state_batch, iter):

        if isinstance(pol, torch.Tensor) or isinstance(val, torch.Tensor):
            state_batch_indices = self.get_state_batch_indices(state_batch,
                                                           iter)

        if isinstance(pol, torch.Tensor):

            policy = torch.sigmoid(pol)[state_batch_indices].reshape(-1, 1)
        else:
            # print(state_batch)
            policy = pol(state_batch)

        if isinstance(val, torch.Tensor):

            state_value = val[state_batch_indices].reshape(-1, 1)

        else:
            # policy = th[i](state)
            state_value = val(state_batch)

        return policy, state_value

    def get_policy_vals_indices_for_iter(self, th, vals, state_batch, iter):
        policies = torch.zeros((self.n_agents, self.batch_size, 1))
        state_values = torch.zeros((self.n_agents, self.batch_size, 1))
        # state_batch_indices = self.get_state_batch_indices(state_batch, iter)
        for i in range(self.n_agents):

            policy, state_value = self.get_policy_and_state_value(th[i],
                                                                  vals[i],
                                                                  state_batch,
                                                                  iter)

            policies[i] = policy
            state_values[i] = state_value

        return policies, state_values

    def get_next_val_history(self, th, vals, val_history, ending_state_batch, iter):
        # The notation and naming here is a bit questionable. Vals is the actual parameterized value function
        # Val_history or state_vals as in some of the other functions are the state values for the given states in
        # some rollout/trajectory

        policies, ending_state_values = self.get_policy_vals_indices_for_iter(
            th, vals, ending_state_batch, iter)

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
        # next_val_history is also based off of the current values (vals)

        # TODO batch this, to be faster and avoid for loops
        # Look through entire code for for loops
        # Finally, move everything to GPU? And then test that also.

        coop_probs = torch.zeros((self.num_iters, self.n_agents, self.batch_size, 1))
        init_state_batch = self.get_init_state_batch()
        state_batch = init_state_batch

        state_vals = torch.zeros(
            (self.num_iters, self.n_agents, self.batch_size, 1))


        for iter in range(self.num_iters):

            policies, state_values, state_batch_indices = self.get_policy_vals_indices_for_iter(
                th, vals, state_batch, iter)

            coop_probs[iter] = policies
            state_vals[iter] = state_values

            state_batch = trajectory[iter].float() # get the next state batch from the trajectory
            state_batch = state_batch.reshape(self.n_agents * self.history_len, self.batch_size)
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

            policies, state_values = self.get_policy_vals_indices_for_iter(th, vals, state_batch, iter)

            policy_history[iter] = policies
            val_history[iter] = state_values

            actions = torch.distributions.binomial.Binomial(probs=policies.detach()).sample()

            curr_step_batch = torch.Tensor(actions)
            # This awkward reshape and transpose stuff gets around some issues with reshaping not preserving the data in the ways I want
            curr_step_batch = curr_step_batch.reshape(self.n_agents, self.batch_size)
            curr_step_batch = curr_step_batch.t()

            # TODO I think the below line doesn't do anything. To confirm
            if not mnist_states:
                state_batch = state_batch.reshape(self.n_agents * self.history_len, self.batch_size)
                state_batch = state_batch.t()



            if self.history_len > 1:
                new_state_batch = torch.zeros_like(state_batch)
                new_state_batch[:, :self.n_agents * (self.history_len-1)] = state_batch[:, self.n_agents:self.n_agents * self.history_len]
                new_state_batch[:, self.n_agents * (self.history_len - 1):] = curr_step_batch
                state_batch = new_state_batch
            else:
                state_batch = curr_step_batch

            if self.using_mnist_states:
                state_batch = self.build_mnist_state_from_classes(state_batch)

            trajectory[iter] = torch.Tensor(actions)

            # Note negative rewards might help with exploration in PG formulation
            total_contrib = sum(actions)
            # total_contrib = actions.sum(dim=0)

            payout_per_agent = total_contrib * self.contribution_factor / self.n_agents
            agent_rewards = -actions + payout_per_agent  # if agent contributed 1, subtract 1, that's what the -actions does
            agent_rewards -= adjustment_to_make_rewards_negative
            rewards[iter] = agent_rewards

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
            # TODO is this detach necessary? Pretty sure it is.



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

        # discounted_vals = val_history * discounts

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


            # # Two way clamp, not yet ppo style
            # log_p_act_or_p_act_ratio = torch.clamp(log_p_act_or_p_act_ratio, min=1 - clip_epsilon, max=1 + clip_epsilon)

            # PPO style clipping
            pos_adv = (advantages > 0).float()
            log_p_act_or_p_act_ratio = pos_adv * torch.minimum(log_p_act_or_p_act_ratio,torch.zeros_like(log_p_act_or_p_act_ratio) + 1+clip_epsilon) + \
                                       (1-pos_adv) * torch.maximum(log_p_act_or_p_act_ratio,torch.zeros_like(log_p_act_or_p_act_ratio) + 1-clip_epsilon)


            # # Below was the weird clipping I was using prior to Sep 22
            # probs_to_clip = (advantages > 0).float()
            #
            # log_p_act_or_p_act_ratio = probs_to_clip * torch.minimum(log_p_act_or_p_act_ratio,torch.zeros_like(log_p_act_or_p_act_ratio) + 1+clip_epsilon) + (1-probs_to_clip) * log_p_act_or_p_act_ratio



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


        deps_up_to_t = (torch.cumsum(sum_over_agents_log_p_act_or_p_act_ratio, dim=0)).reshape(-1, 1, self.batch_size, 1)

        deps_less_than_t = deps_up_to_t - sum_over_agents_log_p_act_or_p_act_ratio.reshape(-1, 1, self.batch_size, 1) # take out the dependency in the given time step

        # Two equivalent formulations - well they should be but somehow the G_t one is wrong.
        # dice_rewards2 is wrong because while the first order gradients match, the higher order ones don't.
        # Look at how the magic box operator works and think about why what was equivalent
        # formulations in the regular policy gradient case is no longer equivalent with the magic box on it
        # p act ratio needs to be producted, not summed, I now think. So this formulation is wrong I think.
        # dice_rewards = (magic_box(torch.cumsum(sum_over_agents_log_p_act_or_p_act_ratio, dim=0)).reshape(-1, 1, self.batch_size, 1) * gamma_t_r_ts).sum(dim=0).mean(dim=1)
        # dice_rewards2 = (magic_box(sum_over_agents_log_p_act_or_p_act_ratio.reshape(-1, 1, self.batch_size, 1)) * G_ts).sum(dim=0).mean(dim=1)
        # loaded_dice_rewards = ((magic_box(deps_up_to_t) - magic_box(deps_less_than_t)) * discounts * R_ts).sum(dim=0).mean(dim=1)
        # loaded_dice_rewards = ((magic_box(deps_up_to_t) - magic_box(deps_less_than_t)) * G_ts).sum(dim=0).mean(dim=1)

        # dice_objective_w_baseline = (magic_box(sum_over_agents_log_p_act_or_p_act_ratio.reshape(-1, 1, self.batch_size, 1)) * advantages).sum(dim=0).mean(dim=1)

        # Look at Loaded DiCE paper to see where this formulation comes from
        loaded_dice_rewards = ((magic_box(deps_up_to_t) - magic_box(deps_less_than_t)) * discounts * advantages).sum(dim=0).mean(dim=1)


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


        # regular_nl_loss = -(torch.cumsum(log_p_act, dim=0) * gamma_t_r_ts).sum(dim=0)

        # return dice_loss, G_ts, regular_nl_loss
        return dice_loss, G_ts, values_loss

        # return dice_loss, G_ts, values_loss, -dice_rewards, -dice_rewards2






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
                 output_size, final_sigmoid=True):
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

        if final_sigmoid:
            layers.append(nn.Sigmoid())

        self.net = nn.Sequential(*layers)


    def forward(self, x):
        output = self.net(x)
        return output


class ConvFC(nn.Module):
    def __init__(self, conv_in_channels, conv_out_channels, input_size, hidden_size, output_size, kernel_size=5, final_sigmoid=True):
        super(ConvFC, self).__init__()

        self.conv_out_channels = conv_out_channels
        # layers = []

        self.layer1 = nn.Conv2d(conv_in_channels, conv_out_channels,
                                kernel_size=kernel_size)
        self.conv_result_size = (
                    input_size - kernel_size + 1)  # no stride or pad here
        self.fc_size = self.conv_result_size ** 2 * self.conv_out_channels
        self.layer2 = nn.Linear(self.fc_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, output_size)

        self.final_sigmoid = final_sigmoid

        # layers.append(self.layer1)
        # layers.append(torch.nn.Tanh())
        # layers.append(self.layer2)
        # layers.append(torch.nn.Tanh())
        # layers.append(self.layer3)
        #
        # if final_sigmoid:
        #     layers.append(nn.Sigmoid())
        # self.net = nn.Sequential(*layers)


    def forward(self, x):

        # assert len(x.shape) >= 3
        # print(x.shape)
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        conv_output = torch.tanh(self.layer1(x))
        # print(conv_output)
        output = conv_output.reshape(-1, self.fc_size)
        # print(output)
        output = torch.tanh(self.layer2(output))
        # print(output)
        output = self.layer3(output)
        # print(output)

        if self.final_sigmoid:
            output = torch.sigmoid(output)


        return output


def init_custom(dims, using_nn=True, nn_hidden_size=16, nn_extra_hidden_layers=0):
    th = []
    f_th = []

    # NN/func approx
    if using_nn:
        for i in range(len(dims)):
            if mnist_states:
                policy_net = ConvFC(conv_in_channels=dims[i],
                                    # mnist specific input is 28x28x1
                                    conv_out_channels=dims[i],
                                    input_size=28,
                                    hidden_size=nn_hidden_size,
                                    output_size=1,
                                    final_sigmoid=True)
            else:
                policy_net = NeuralNet(input_size=dims[i], hidden_size=nn_hidden_size, extra_hidden_layers=nn_extra_hidden_layers,
                              output_size=1)
            f_policy_net = higher.patch.monkeypatch(policy_net, copy_initial_weights=True,
                                     track_higher_grads=True)
            # print(f_policy_net)

            th.append(policy_net)
            f_th.append(f_policy_net)

            # th.append(f_policy_net)

    # Tabular policies
    else:
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

    optims_th = construct_optims(th, lr_policies)
    # optims = None

    assert len(th) == len(dims)
    # assert len(optims) == len(dims)

    vals = []
    f_vals = []

    for i in range(len(dims)):
        if using_nn:
            if mnist_states:
                vals_net = ConvFC(conv_in_channels=dims[i],
                                  # mnist specific input is 28x28x1
                                  conv_out_channels=dims[i],
                                  input_size=28,
                                  hidden_size=nn_hidden_size,
                                  output_size=1,
                                  final_sigmoid=False)
            else:
                vals_net = NeuralNet(input_size=dims[i], hidden_size=nn_hidden_size,
                                  extra_hidden_layers=nn_extra_hidden_layers,
                                  output_size=1, final_sigmoid=False)
            vals.append(vals_net)
            f_vals_net = higher.patch.monkeypatch(vals_net,
                                                    copy_initial_weights=True,
                                                    track_higher_grads=True)
            # print(f_policy_net)

            f_vals.append(f_vals_net)
        else:
            vals.append(torch.nn.init.normal_(torch.empty(2**n_agents + 1, requires_grad=True), std=0.1))


    assert len(vals) == len(dims)

    optims_vals = construct_optims(vals, lr_values)

    return th, optims_th, vals, optims_vals, f_th, f_vals

def construct_diff_optims(th_or_vals, lrs, f_th_or_vals):
    optims = []
    for i in range(len(th_or_vals)):
        if not isinstance(th_or_vals[i], torch.Tensor):
            optim = torch.optim.SGD(th_or_vals[i].parameters(), lr=lrs[i])

            diffoptim = higher.get_diff_optim(optim, th_or_vals[i].parameters(), f_th_or_vals[i])
            optims.append(diffoptim)

        else:
            # Don't use for now
            optim = torch.optim.SGD([th_or_vals[i]], lr=lrs[i])
            # print(th_or_vals)
            # print(f_th_or_vals)
            # diffoptim = higher.get_diff_optim(optim, [th_or_vals[i]])
            diffoptim = higher.optim.DifferentiableSGD(optim, [th_or_vals[i]])
            # diffoptim = higher.get_diff_optim(optim, [th_or_vals[i]], f_th_or_vals[i])
            optims.append(diffoptim)
    return optims

def construct_optims(th_or_vals, lrs):
    optims = []
    for i in range(len(th_or_vals)):
        if not isinstance(th_or_vals[i], torch.Tensor):
            optim = torch.optim.SGD(th_or_vals[i].parameters(), lr=lrs[i])
            optims.append(optim)
        else:
            # Don't use for now
            optim = torch.optim.SGD([th_or_vals[i]], lr=lrs[i])
            optims.append(optim)
    return optims

def get_gradient(function, param):
    grad = torch.autograd.grad(function, param, create_graph=True)[0]
    return grad

def print_policy_info(policy, i):

    print("Policy {}".format(i))
    print(
        "(Probabilities are for cooperation/contribution, for states 00...0 (no contrib,..., no contrib), 00...01 (only last player contrib), 00...010, 00...011, increasing in binary order ..., 11...11 , start)")

    print(policy)



def print_additional_policy_info(G_ts, discounted_sum_of_adjustments, truncated_coop_payout, inf_coop_payout):
    print("Discounted Sum Rewards (Avg over batches) in this episode (removing negative adjustment): ")

    print(G_ts[0].mean(dim=1).reshape(-1) + discounted_sum_of_adjustments)
    print("Max Avg Coop Payout (Truncated Horizon): {:.3f}".format(
        truncated_coop_payout))
    print("Max Avg Coop Payout (Infinite Horizon): {:.3f}".format(
        inf_coop_payout))


def build_all_combs_state_batch(n):
    state_batch = torch.cat((build_bin_matrix(n, 2 ** n),
                             torch.Tensor([init_state_representation] * n).reshape(1, -1)))
    return state_batch


def print_value_info(vals, i, mnist_states=False, contrib_game=None):
    print("Values {}".format(i))
    if isinstance(vals[i], torch.Tensor):
        values = vals[i]
    else:
        state_batch = build_all_combs_state_batch(n_agents * args.history_len)
        if mnist_states:
            assert contrib_game is not None
            state_batch = contrib_game.build_mnist_state_from_classes(state_batch)

        values = vals[i](state_batch)
    print(values)


def print_policies_for_all_states(th, mnist_states=False, contrib_game=None):
    for i in range(len(th)):
        if isinstance(th[i], torch.Tensor):
            policy = torch.sigmoid(th[i])

        else:
            state_batch = build_all_combs_state_batch(n_agents * args.history_len)
            if mnist_states:
                assert contrib_game is not None
                state_batch = contrib_game.build_mnist_state_from_classes(
                    state_batch)

            policy = th[i](state_batch)

        print_policy_info(policy, i)



def update_th(th, gradient_terms_or_Ls, lr_policies, eta, algos, using_samples):
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
    parser.add_argument("--etas", nargs="+", type=float, default=[20, 12],
                        help="list of etas to try")
    parser.add_argument("--lr_policies", type=float, default=0.05,
                        help="same learning rate across all policies for now")
    parser.add_argument("--lr_values_scale", type=float, default=0.3,
                        help="scale lr_values relative to lr_policies")
    parser.add_argument("--inner_steps", type=int, default=1)
    parser.add_argument("--using_nn", action="store_true",
                        help="use neural net/func approx instead of tabular policy")
    parser.add_argument("--nn_hidden_size", type=int, default=16)
    parser.add_argument("--nn_extra_layers", type=int, default=0)
    parser.add_argument("--set_seed", action="store_true",
                        help="set manual seed")
    parser.add_argument("--seed", type=int, default=1, help="for seed")
    parser.add_argument("--extra_value_updates", type=int, default=0,
                        help="additional value function updates (0 means just 1 update per outer rollout)")
    parser.add_argument("--history_len", type=int, default=1, help="Number of steps lookback that each agent gets as state")
    parser.add_argument("--mnist_states", action="store_true",
                        help="use MNIST digits as state representation")
    parser.add_argument("--init_state_representation", type=int, default=2)

    args = parser.parse_args()


    init_state_representation = args.init_state_representation

    if args.set_seed:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

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

    if args.history_len > 1:
        assert args.using_nn # Right now only supported for func approx.

    # # Why does LOLA agent sometimes defect at start but otherwise play TFT? Policy gradient issue?
    # etas = [
    #     0.01 * 5]  # wait actually this doesn't seem to work well at all... no consistency in results without dice... is it because we missing 1 term? this is batch size 1
    # if not using_samples:
    #     etas = [0.05 * 20, 0.05 * 12]

    mnist_states = args.mnist_states


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

        if batch_size == n_agents or batch_size == rollout_len or rollout_len == n_agents:
            raise Exception("Having two of batch size, rollout len, or n_agents equal will cause insidious bugs when reshaping dimensions")
            # TODO refactor/think about a way to avoid these bugs

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

                    game = ContributionGame(n=n_agents, gamma=gamma,
                                            batch_size=batch_size,
                                            num_iters=rollout_len,
                                            contribution_factor=contribution_factor,
                                            contribution_scale=contribution_scale,
                                            history_len=args.history_len,
                                            using_mnist_states=mnist_states)
                    dims = game.dims

                    th, optims_th, vals, optims_vals, f_th, f_vals = init_custom(dims, args.using_nn)

                    # I think part of the issue is if policy saturates at cooperation it never explores and never tries defect
                    # How does standard reinforce/policy gradient get around this? Entropy regularization
                    # Baseline might help too. As might negative rewards everywhere.


                if using_DiCE:
                    inner_steps = [args.inner_steps] * n_agents
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
                            if args.using_nn:
                                static_th_copy = th
                                static_vals_copy = vals
                            else:
                                static_th_copy = copy.deepcopy(th)
                                static_vals_copy = copy.deepcopy(vals)


                            for i in range(n_agents):
                                K = inner_steps[i]

                                # TODO later confirm that this deepcopy is working properly for NN also
                                if args.using_nn:
                                    theta_primes = static_th_copy
                                    val_primes = static_vals_copy # should work if it is functional/stateless
                                else:
                                    theta_primes = copy.deepcopy(static_th_copy)
                                    val_primes = copy.deepcopy(static_vals_copy)


                                f_th_primes = []
                                if args.using_nn:
                                    for ii in range(n_agents):
                                        f_th_primes.append(higher.patch.monkeypatch(theta_primes[ii], copy_initial_weights=True, track_higher_grads=True))
                                optims_th_primes = construct_diff_optims(theta_primes, lr_policies * eta, f_th_primes)
                                f_vals_primes = []
                                if args.using_nn:
                                    for ii in range(n_agents):
                                        f_vals_primes.append(
                                            higher.patch.monkeypatch(
                                                val_primes[ii],
                                                copy_initial_weights=True,
                                                track_higher_grads=True))
                                optims_vals_primes = construct_diff_optims(
                                    val_primes, lr_policies * eta,
                                    f_vals_primes)

                                if args.using_nn:
                                    mixed_thetas = f_th_primes
                                    mixed_vals = f_vals_primes
                                    mixed_thetas[i] = th[i]
                                    mixed_vals[i] = vals[i]
                                else:
                                    mixed_thetas = theta_primes
                                    mixed_vals = val_primes
                                    mixed_thetas[i] = th[i]
                                    mixed_vals[i] = vals[i]

                                # print(mixed_thetas)
                                if eta != 0:

                                    if repeat_train_on_same_samples:
                                        trajectory, rewards, policy_history, val_history, next_val_history = game.rollout(
                                            mixed_thetas, mixed_vals)
                                        for step in range(K):
                                            if step == 0:
                                                dice_loss, _, values_loss = game.get_dice_loss(trajectory, rewards,
                                                    policy_history, val_history, next_val_history, old_policy_history=policy_history)

                                            else:
                                                new_policies, new_vals, next_new_vals = game.get_policies_vals_for_states(
                                                    mixed_thetas, mixed_vals, trajectory)
                                                # Using the new policies and vals now
                                                # Always be careful not to overwrite/reuse names of existing variables
                                                dice_loss, _, values_loss = game.get_dice_loss(
                                                    trajectory, rewards, new_policies, new_vals, next_new_vals, old_policy_history=policy_history)

                                            grads = [None] * n_agents

                                            # Note only policy update, no value update here
                                            # because we are updating off policy
                                            # We need to keep the advantage consistent with the policy that
                                            # collected the data we are training on
                                            # So no value update in between policy updates on the same data
                                            for j in range(n_agents):
                                                if j != i:
                                                    if isinstance(mixed_thetas[j], torch.Tensor):
                                                        # Higher with diffopt on the tensor can work too
                                                        # I think what needs to be done is a simpler formulation
                                                        # Where you just construct the diffopt, no fmodel stuff on the tensor
                                                        # and then directly use that diffopt. I had it working before

                                                        # print(mixed_thetas[j])

                                                        grad = get_gradient(
                                                            dice_loss[j],
                                                            mixed_thetas[j])
                                                        mixed_thetas[j] = mixed_thetas[j] - lr_policies[j] * eta * grad  # This step is critical to allow the gradient to flow through
                                                        # You cannot use torch.no_grad on this step

                                                        # print(mixed_thetas[j])

                                                    else:

                                                        optim_update(optims_th_primes[j],
                                                            dice_loss[j], mixed_thetas[j].parameters())

                                            # for j in range(n_agents):
                                            #     if j != i:
                                            #
                                            #         if isinstance(mixed_thetas[j],
                                            #                       torch.Tensor):
                                            #             grad = get_gradient(
                                            #                 dice_loss[j],
                                            #                 mixed_thetas[j])
                                            #             grads[j] = grad
                                            #
                                            #         else:
                                            #             1/0
                                            # for j in range(n_agents):
                                            #     # Need a separate loop to deal with issue where you can't update differentiably first in p2 otherwise p3 will see that and diff through that
                                            #     if j != i:
                                            #         if isinstance(mixed_thetas[j],
                                            #                       torch.Tensor):
                                            #
                                            #             mixed_thetas[j] = \
                                            #             mixed_thetas[j] - \
                                            #             lr_policies[
                                            #                 j] * eta * grads[
                                            #                 j]  # This step is critical to allow the gradient to flow through
                                            #         else:
                                            #             1 / 0

                                            # Also TODO Aug 23 is do an outer loop with number of steps also


                                            # print("---Agent {} Rollout {}---".format(i, step))
                                            # print_policies_for_all_states(
                                                # mixed_thetas)


                                    else: # no repeat train on samples, this is the original DiCE formulation

                                        for step in range(K):

                                            trajectory, rewards, policy_history, val_history, next_val_history = game.rollout(
                                                mixed_thetas, mixed_vals)
                                            dice_loss, _, values_loss = game.get_dice_loss(
                                                trajectory,
                                                rewards,
                                                policy_history, val_history, next_val_history)

                                            grads = [None] * n_agents
                                            # grad_vals = [None] * n_agents # For stability don't do this

                                            for j in range(n_agents):
                                                if j != i:
                                                    if isinstance(mixed_thetas[j], torch.Tensor):
                                                        # Higher with diffopt on the tensor can work too
                                                        # I think what needs to be done is a simpler formulation
                                                        # Where you just construct the diffopt, no fmodel stuff on the tensor
                                                        # and then directly use that diffopt. I had it working before

                                                        # print(mixed_thetas[j])

                                                        grad = get_gradient(
                                                            dice_loss[j],
                                                            mixed_thetas[j])
                                                        mixed_thetas[j] = mixed_thetas[j] - lr_policies[j] * eta * grad  # This step is critical to allow the gradient to flow through
                                                        # You cannot use torch.no_grad on this step

                                                        # print(mixed_thetas[j])

                                                        # TODO The regular DiCE formulation should have value updates on this inner loop too
                                                        # However either 1) it is hard to choose the right hyperparameter
                                                        # or 2) you end up using small value updates which don't do much
                                                        # or 3) you require multiple value update iterations which is going to require more time/make training slower

                                                    else:

                                                        optim_update(optims_th_primes[j],
                                                            dice_loss[j], mixed_thetas[j].parameters())

                                            # print("---Agent {} Rollout {}---".format(i, step))
                                            # print_policies_for_all_states(mixed_thetas)


                                # Now calculate outer step using for each player a mix of the theta_primes and old thetas

                                trajectory, rewards, policy_history, val_history, next_val_history = game.rollout(
                                    mixed_thetas, mixed_vals)


                                if repeat_train_on_same_samples:
                                    dice_loss, G_ts, values_loss = game.get_dice_loss(
                                        trajectory, rewards,
                                        policy_history, val_history, next_val_history,
                                        old_policy_history=policy_history)
                                else:
                                    dice_loss, G_ts, values_loss = game.get_dice_loss(
                                        trajectory, rewards,
                                        policy_history, val_history, next_val_history)

                                # print("---Agent {} Rollout---".format(i))
                                # print_policies_for_all_states(mixed_thetas)
                                # print_additional_policy_info(G_ts, discounted_sum_of_adjustments, truncated_coop_payout, inf_coop_payout)


                                # NOTE: TODO potentially: G_ts here may not be the best choice
                                # But it should be close enough to give an idea of what the rewards roughly look like
                                # The reason why G_ts may not be best representation is I believe the way
                                # I have the calculation set up now, it only takes G_ts from the last agent's rollout
                                # Because of asynchronous rollouts, it's weird, we would have to have one more rollout at the end
                                # if we just wanted to evaluate what the newly updated policies look like when played together.

                                # TODO note mixed_th[i] should be == th[i] here
                                if isinstance(th[i], torch.Tensor):
                                    # optim_update(optims_th[i], dice_loss[i],
                                    #              [th[i]])
                                    # optim_update(optims_vals[i],
                                    #              values_loss[i],
                                    #              [vals[i]])

                                    grad = get_gradient(dice_loss[i], th[i])
                                    grad_val = get_gradient(values_loss[i], vals[i])


                                    with torch.no_grad():
                                        th[i] -= lr_policies[i] * grad
                                        vals[i] -= lr_values[i] * grad_val
                                        # th[i] -= lr_policies[i] * (b-a)

                                    # TODO Be careful with +/- formulation now...
                                    # TODO rename the non-DiCE terms as rewards
                                    # And keep the DiCE terms as losses


                                else:
                                    optim_update(optims_th[i], dice_loss[i])
                                    optim_update(optims_vals[i], values_loss[i])

                                    # optim_update(optims_th[i], dice_loss[i], f_th[i].parameters())
                                    # optim_update(optims_vals[i], values_loss[i], f_vals[i].parameters())
                                    #
                                    # copyNN(th[i], f_th[i])
                                    # copyNN(vals[i], f_vals[i])

                                for _ in range(args.extra_value_updates):
                                    _, val_history, next_val_history = game.get_policies_vals_for_states(mixed_thetas, mixed_vals, trajectory)

                                    if repeat_train_on_same_samples:
                                        dice_loss, G_ts, values_loss = game.get_dice_loss(
                                            trajectory, rewards,
                                            policy_history, val_history,
                                            next_val_history,
                                            old_policy_history=policy_history)
                                    else:
                                        dice_loss, G_ts, values_loss = game.get_dice_loss(
                                            trajectory, rewards,
                                            policy_history, val_history,
                                            next_val_history)

                                    if isinstance(vals[i], torch.Tensor):
                                        grad_val = get_gradient(values_loss[i],
                                                                vals[i])
                                        with torch.no_grad():
                                            vals[i] -= lr_values[i] * grad_val
                                    else:
                                        optim_update(optims_vals[i],
                                                     values_loss[i])




                        else:

                            trajectory, rewards, policy_history = game.rollout(th)
                            gradient_terms = game.get_gradient_terms(trajectory, rewards, policy_history)

                            th, losses, G_ts, nl_terms, lola_terms, grad_2_return_1 = \
                                update_th(th, gradient_terms, lr_policies, eta, algos, using_samples=using_samples)
                    else:
                        th, losses, G_ts, nl_terms, lola_terms, grad_2_return_1 = \
                            update_th(th, Ls, lr_policies, eta, algos, using_samples=using_samples)


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
                            print_additional_policy_info(G_ts,
                                                         discounted_sum_of_adjustments,
                                                         truncated_coop_payout,
                                                         inf_coop_payout)
                            print_policies_for_all_states(th, mnist_states, game)
                            for i in range(n_agents):
                                print_value_info(vals, i, mnist_states, game)
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

