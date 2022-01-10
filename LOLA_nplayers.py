import numpy as np
import torch
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F

import higher
# TODO credit the higher repo (and let authors know - have link to paper)

import datetime

import copy

import argparse

import random

from timeit import default_timer as timer



def simple_fwd_solver(f, init_point, precision=1e-5):
    prev_point, curr_point = init_point, f(init_point)
    while np.linalg.norm(prev_point - curr_point) > precision:
        prev_point, curr_point = curr_point, f(curr_point)
    return curr_point


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
    # p = torch.zeros(size)
    p = torch.prod(bin_mat * pc + (1 - bin_mat) * pd, dim=1)
    return p


def magic_box(x):
    return torch.exp(x - x.detach())


def copyNN(copy_to_net, copy_from_net):
    copy_to_net.load_state_dict(copy_from_net.state_dict())

def optim_update(optim, loss, params=None):
    if params is not None:
        #diffopt step here
        return optim.step(loss, params)
    else:
        optim.zero_grad()
        loss.backward(retain_graph=True)
        optim.step()


def reverse_cumsum(x, dim):
    return x + torch.sum(x, dim=dim, keepdims=True) - torch.cumsum(x, dim=dim)

def ipdn(n=2, gamma=0.96, contribution_factor=1.6, contribution_scale=False):
    # This is an exact gradient setting - constructing the transition probability matrix and doing inverse
    # Exact gradient I use to mean "we have access to the transition model and can therefore calculate the exact value function given policies"
    # rather than being forced to do rollouts in the case of
    # See the LOLA full paper appendix A.2 for derivations here
    # This part (and many of the other exact gradient parts) are adapted from https://github.com/aletcher/stable-opponent-shaping
    # See also the comments in ContributionGame. This is the same formulation of the contribution game which is a generalization of the IPD

    dims = [2 ** n + 1 for _ in range(n)]
    state_space = dims[0]

    if contribution_scale:
        contribution_factor = contribution_factor * n
    else:
        assert contribution_factor > 1

    bin_mat = build_bin_matrix(n, size=state_space - 1)

    payout_vectors = torch.zeros((n, state_space - 1))  # one vector for each player, state space - 1 because one is the initial state. This is the r^1 or r^2 in the LOLA paper exact gradient formulation. In the 2p case this is for DD, DC, CD, CC

    for agent in range(n):
        for state in range(state_space - 1):
            l = bin_inttensor_from_int(state, n)
            total_contrib = sum(l)
            agent_payout = total_contrib * contribution_factor / n - l[
                agent]  # if agent contributed 1, subtract 1
            agent_payout -= adjustment_to_make_rewards_negative
            payout_vectors[agent][state] = agent_payout

    def Ls(th):
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

        init_pc = torch.zeros(n)
        for i in range(n):

            if init_state_representation == 1:
                if args.ill_condition:
                    p_i_0 = torch.sigmoid((ill_cond_matrices[i] @ th[i])[-2])
                else:
                    p_i_0 = torch.sigmoid(th[i][-2]) # force all coop at the beginning in this special case
            else:
                if args.ill_condition:
                    p_i_0 = torch.sigmoid((ill_cond_matrices[i] @ th[i])[-1])
                else:
                    p_i_0 = torch.sigmoid(th[i][-1])  # start state is at the end; this is a (kind of arbitrary) design choice

            init_pc[i] = p_i_0
            # Policy represents prob of coop (taking action 1)

        p = build_p_vector(n, state_space - 1, init_pc, bin_mat)

        # This part and the below part might be optimizable (without a loop) but it doesn't seem trivial
        # Probabilities in the states other than the start state
        all_p_is = torch.zeros((n, state_space - 1))
        for i in range(n):
            if args.ill_condition:
                p_i = torch.sigmoid((ill_cond_matrices[i] @ th[i])[0:-1])
            else:
                p_i = torch.sigmoid(th[i][0:-1])
            all_p_is[i] = p_i

        # Transition Matrix
        # Remember now our transition matrix top left is DDD...D
        # 0 is defect in this formulation, 1 is contributing/cooperating
        P = torch.zeros((state_space - 1, state_space - 1))
        for curr_state in range(state_space - 1):
            i = curr_state
            pc = all_p_is[:, i]
            p_new = build_p_vector(n, state_space - 1, pc, bin_mat)
            P[i] = p_new

        # Here instead of using infinite horizon which is the implicit assumption in the derivation from the previous parts (see again the LOLA appendix A.2)
        # We can consider a finite horizon and literally just unroll the calculation
        # You could probably just do 2 inverses and do some multiplication by discount factor
        # and subtract instead of doing this loop but oh well, this is more for testing and comparing anyway.
        if args.exact_finite_horizon:
            gamma_t_P_t = torch.eye(state_space - 1)
            running_total = torch.eye(state_space - 1)

            for t in range(1, args.rollout_len):
                gamma_t_P_t = gamma * gamma_t_P_t @ P
                running_total += gamma_t_P_t
            M = p @ gamma_t_P_t
        else:
            M = torch.matmul(p, torch.inverse(torch.eye(state_space - 1) - gamma * P))

        # Remember M is just the steady state probabilities for each of the states
        # It is a vector, not a matrix.

        L_all = []
        for i in range(n):
            payout_vec = payout_vectors[i]
            L_i = torch.matmul(M, payout_vec)
            L_all.append(L_i)

        return L_all

        # IMPORTANT NOTE: Right now these are positive values (e.g. rewards) rather than losses, which is somewhat confusing
        # Therefore these are not Ls (losses) but rather rewards, and the loss must be the negative reward
        # So must be careful in using optimizers
        # I made them positive so I could calculate gradient ascent
        # Anyway the signs are all consistent, but you have to be careful when modifying this code and when backpropagating the losses (negative rewards)

    return dims, Ls


class Game():
    def __init__(self, n, init_state_representation, history_len=1,  state_type='one_hot'):
        self.n_agents = n
        self.state_type = state_type
        self.history_len = history_len
        self.init_state_representation = init_state_representation
        if args.ill_condition:
            self.dd_stretch_factor = args.dd_stretch_factor #30
            self.all_state_stretch_factor = args.all_state_stretch_factor #0.1
            # So then dd_stretch_factor * all state stretch is what you get in the DD state

    def print_policy_info(self, policy, i):
        print("Policy {}".format(i+1))
        # print("(Probabilities are for cooperation/contribution, for states 00...0 (no contrib,..., no contrib), 00...01 (only last player contrib), 00...010, 00...011, increasing in binary order ..., 11...11 , start)")
        print(policy)

    def print_reward_info(self, G_ts, discounted_sum_of_adjustments,
                                     truncated_coop_payout, inf_coop_payout, env):

        print("Discounted Sum Rewards (Avg over batches) in this episode (removing negative adjustment): ")
        print(G_ts[0].mean(dim=1).reshape(-1) + discounted_sum_of_adjustments)

        if env == 'ipd':
            print("Max Avg Coop Payout (Truncated Horizon): {:.3f}".format(
                truncated_coop_payout))
            print("Max Avg Coop Payout (Infinite Horizon): {:.3f}".format(
                inf_coop_payout))

    def build_all_combs_state_batch(self):

        if self.state_type == 'majorTD4':
            dim = 2 * args.history_len
        else:
            dim = self.n_agents * self.history_len

        state_batch = torch.cat((build_bin_matrix(dim, 2 ** dim),
                                 torch.Tensor([init_state_representation] * dim).reshape(1, -1)))

        if self.state_type == 'mnist':
            state_batch = self.build_mnist_state_from_classes(state_batch)
        elif self.state_type == 'one_hot':
            state_batch = self.build_one_hot_from_batch(state_batch.t(),
                                                        self.action_repr_dim,
                                                                one_at_a_time=False)
        elif self.state_type == 'majorTD4':
            state_batch = self.build_one_hot_from_batch(state_batch,
                                                        self.action_repr_dim,
                                                                one_at_a_time=False,
                                                                simple_2state_build=True)
        return state_batch

    def print_value_info(self, vals, agent_num_i):
        i = agent_num_i
        print("Values {}".format(i+1))
        if isinstance(vals[i], torch.Tensor):
            values = vals[i]
        else:
            state_batch = self.build_all_combs_state_batch()
            values = vals[i](state_batch)
        print(values)

    def print_values_for_all_states(self, vals):
        for i in range(len(vals)):
            self.print_value_info(vals, i)

    def get_policy_for_all_states(self, th, i):
        if isinstance(th[i], torch.Tensor):
            if args.ill_condition:
                ill_cond_policy = torch.sigmoid(ill_cond_matrices[i] @ th[i])
            policy = torch.sigmoid(th[i])

        else:
            state_batch = self.build_all_combs_state_batch()
            # print(state_batch)
            # if contrib_game.using_mnist_states :
            #     state_batch = contrib_game.build_mnist_state_from_classes(
            #         state_batch)
            if args.ill_condition:
                simple_state_repr_batch = self.one_hot_to_simple_repr(
                    state_batch)
                # TODO MAKE THIS NOT HARD CODED

                simple_mask = (simple_state_repr_batch.sum(
                    dim=-1) == 0).unsqueeze(-1)  # DD state

                ill_cond_policy = torch.sigmoid(
                    th[i](state_batch) * (self.all_state_stretch_factor) * (
                                (self.dd_stretch_factor - 1) * simple_mask + 1))
                # print(ill_cond_policy)
                # really ugly but basically the goal is simple mask is 30 x where DD state is and 1 elsewhere

            policy = torch.sigmoid(th[i](state_batch))

        if args.ill_condition:
            policy = ill_cond_policy

        return policy


    def get_policies_for_all_states(self, th):
        policies = []
        for i in range(len(th)):
            policy = self.get_policy_for_all_states(th, i)
            policies.append(policy)
        return policies

    def print_policies_for_all_states(self, th):
        for i in range(len(th)):
            policy = self.get_policy_for_all_states(th, i)
            self.print_policy_info(policy, i)
            # if args.ill_condition:
            #     self.print_policy_info(ill_cond_policy, i)

    def print_policy_and_value_info(self, th, vals):
        self.print_policies_for_all_states(th)
        self.print_values_for_all_states(vals)




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
    def __init__(self, n, batch_size, num_iters, gamma=0.96, contribution_factor=1.6,
                 contribution_scale=False, history_len=1, state_type='one_hot'):

        super().__init__(n, init_state_representation=args.init_state_representation, history_len=history_len, state_type=state_type)

        self.gamma = gamma
        self.contribution_factor = contribution_factor
        self.contribution_scale = contribution_scale
        self.batch_size = batch_size
        self.num_iters = num_iters

        if self.state_type == 'one_hot' or self.state_type == 'majorTD4':
            self.action_repr_dim = 3  # one hot with 3 dimensions, dimension 0 for defect, 1 for contrib/coop, 2 for start
        else:
            self.action_repr_dim = 1  # a single dimensional observation that can take on different vales e.g. 0, 1, init_state_repr

        if self.state_type == 'majorTD4':
            # Following the Barbosa 2020 paper. always 2 because, 1 state for majority coop/defect, 1 for past last action
            self.dims = [2 * history_len * self.action_repr_dim] * n
        else:
            self.dims = [n * history_len * self.action_repr_dim] * n
        """
        for dims, the last n is the number of agents, basically dims[i] is the dim for each agent
        It's sort of a silly way to set things up in the event that all agents are the same
        which is what I am currently doing for all of my experiments
        but would make sense if you mix agents (e.g. one agent has the MajorTD4 state, one has an MNIST state, etc.)
        But I am not sure why you would want to mix the agents like that (giving
        different agents different vision/observations of the same underlying state, essentially)
        """

        if self.contribution_scale:
            self.contribution_factor = contribution_factor * n
        else:
            assert self.contribution_factor > 1

        self.dec_value_mask = (2 ** torch.arange(n - 1, -1, -1)).float()

        if self.state_type == 'mnist':
            from torchvision import datasets, transforms

            mnist_train = datasets.MNIST('data', train=True, download=True,
                                         transform=transforms.ToTensor())
            self.coop_class = args.mnist_coop_class
            self.defect_class = args.mnist_defect_class
            idx_coop = (mnist_train.targets) == self.coop_class
            idx_defect = (mnist_train.targets) == self.defect_class
            idx_init = (mnist_train.targets) == init_state_representation # Here suppose you use 2, then you will have digit class 2 for the init state

            self.mnist_coop_class_dset = torch.utils.data.dataset.Subset(
                mnist_train, np.where(idx_coop == 1)[0])
            self.mnist_defect_class_dset = torch.utils.data.dataset.Subset(
                mnist_train, np.where(idx_defect == 1)[0])
            self.mnist_init_class_dset = torch.utils.data.dataset.Subset(
                mnist_train, np.where(idx_init == 1)[0])
            self.len_mnist_coop_dset = len(self.mnist_coop_class_dset)
            self.len_mnist_defect_dset = len(self.mnist_defect_class_dset)
            self.len_mnist_init_dset = len(self.mnist_init_class_dset)

    def build_mnist_state_from_classes(self, batch_tensor):
        batch_tensor_dims = batch_tensor.shape

        mnist_state = torch.zeros((batch_tensor_dims[0], batch_tensor_dims[1], 28, 28))
        # Should try to optimize/vectorize
        for b in range(batch_tensor_dims[0]):
            for c in range(batch_tensor_dims[1]):
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
        if self.state_type == 'mnist':
            integer_state_batch = torch.ones(
                (self.batch_size,
                 self.n_agents * self.history_len)) * init_state_representation
            init_state_batch = self.build_mnist_state_from_classes(integer_state_batch)

        elif self.state_type == 'one_hot':
            # Note that in the 1 hot state representation, class 0 is defect (0 contribution),
            # class 1 is cooperate (1 contribution)
            # class 2 is start state (unused (i.e. always 0) if initializing to coop in the first state (init_state_representation 1))
            init_state_batch = torch.zeros(
                (self.batch_size,
                 self.n_agents * self.history_len, self.action_repr_dim))
            init_state_batch[:,:,init_state_representation] += 1
            init_state_batch = init_state_batch.reshape(self.batch_size, self.n_agents * self.history_len * self.action_repr_dim)

        elif self.state_type == 'majorTD4':
            init_state_batch = torch.zeros(
                (self.n_agents, self.batch_size,
                 2 * self.history_len, self.action_repr_dim)) # additional self.n_agents at the beginning because we need different obs for different agents here
            init_state_batch[:, :, :, init_state_representation] += 1
            init_state_batch = init_state_batch.reshape(self.n_agents, self.batch_size, 2 * self.history_len * self.action_repr_dim)
            # So then here this is not really a state batch, but more of an observation batch

        else: # old / only for tabular, just 0, 1, or 2 for the state
            init_state_batch = torch.ones(
                (self.batch_size, self.n_agents * self.history_len)) * init_state_representation

        return init_state_batch

    def int_from_bin_inttensor(self, bin_tens):
        return torch.sum(self.dec_value_mask * bin_tens, -1).item()

    def get_state_batch_indices(self, state_batch, iter):
        if iter == 0:
            # we just started
            assert self.state_type == 'old'
            assert state_batch[0][0] - init_state_representation == 0
            indices = [-1] * self.batch_size
        else:
            indices = list(map(self.int_from_bin_inttensor, state_batch))
        return indices

    def get_policy_and_state_value(self, pol, val, i, state_batch, iter):

        if isinstance(pol, torch.Tensor) or isinstance(val, torch.Tensor):
            state_batch_indices = self.get_state_batch_indices(state_batch,
                                                           iter)

        if isinstance(pol, torch.Tensor):
            if args.ill_condition:
                policy = torch.sigmoid(ill_cond_matrices[i] @ pol)[state_batch_indices].reshape(-1, 1)
            else:
                policy = torch.sigmoid(pol)[state_batch_indices].reshape(-1, 1)
        else:
            if args.ill_condition:
                simple_state_repr_batch = self.one_hot_to_simple_repr(state_batch)
                # Essentially replicating the ill_conditioning in the exact case
                simple_mask = (simple_state_repr_batch.sum(dim=-1) == 0).unsqueeze(-1) # DD state
                policy = torch.sigmoid(pol(state_batch) * (self.all_state_stretch_factor) * ((self.dd_stretch_factor - 1) * simple_mask + 1))
                # really ugly but the goal is simple mask is 30x (or dd stretch factor times) where DD is and 1 elsewhere
            else:
                policy = torch.sigmoid(pol(state_batch))

        if isinstance(val, torch.Tensor):
            state_value = val[state_batch_indices].reshape(-1, 1)

        else:
            state_value = val(state_batch)

        return policy, state_value

    def get_policy_vals_indices_for_iter(self, th, vals, state_batch, iter):
        policies = torch.zeros((self.n_agents, self.batch_size, 1))
        state_values = torch.zeros((self.n_agents, self.batch_size, 1))
        for i in range(self.n_agents):

            if self.state_type == 'majorTD4':
                # Different obs for each agent
                policy, state_value = self.get_policy_and_state_value(th[i],
                                                                      vals[i], i,
                                                                      state_batch[i],
                                                                      iter)

            else:
                # same state batch for all agents
                policy, state_value = self.get_policy_and_state_value(th[i],
                                                                      vals[i], i,
                                                                      state_batch,
                                                                      iter)


            policies[i] = policy
            state_values[i] = state_value

        return policies, state_values

    def get_next_val_history(self, th, vals, val_history, ending_state_batch, iter):
        # My notation and naming here is a bit questionable. Sorry. Vals is the actual parameterized value function
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

    def get_policies_vals_for_states(self, th, vals, obs_history):
        # Returns coop_probs and state_vals, which are the equivalent of
        # policy_history and val_history, except they are using the current policies and values
        # (which may be different from the policies and values that were originally
        # used to rollout in the environment)
        # next_val_history is also based off of the current values (vals)

        coop_probs = torch.zeros((self.num_iters, self.n_agents, self.batch_size, 1))
        init_state_batch = self.get_init_state_batch()
        state_batch = init_state_batch

        state_vals = torch.zeros(
            (self.num_iters, self.n_agents, self.batch_size, 1))

        for iter in range(self.num_iters):

            policies, state_values = self.get_policy_vals_indices_for_iter(
                th, vals, state_batch, iter)

            coop_probs[iter] = policies
            state_vals[iter] = state_values

            state_batch = obs_history[iter].float() # get the next state batch from the state history

        next_val_history = self.get_next_val_history(th, vals, state_vals, state_batch,
                                                     iter + 1)

        return coop_probs, state_vals, next_val_history


    def one_hot_to_simple_repr(self, one_hot_batch):
        simple_repr_tensor = torch.zeros((one_hot_batch.shape[0], self.n_agents))
        # No history len > 1 supported here
        start = 0
        end = self.action_repr_dim

        index_collector = torch.zeros((one_hot_batch.shape[0], self.action_repr_dim))
        for a in range(self.action_repr_dim):
            index_collector[:, a] += a

        for i in range(self.n_agents):
            simple_repr_tensor[:,i] = (one_hot_batch[:,start:end] * index_collector).sum(dim=-1)
            start += self.action_repr_dim
            end += self.action_repr_dim

        return simple_repr_tensor


    def build_one_hot_from_batch(self, curr_step_batch, one_hot_dim, one_at_a_time=True, range_end=None, simple_2state_build=False):

        if range_end is None:
            range_end = self.n_agents
        curr_step_batch_one_hot = torch.nn.functional.one_hot(
            curr_step_batch.long(), one_hot_dim).squeeze(dim=2)

        if simple_2state_build:
            new_tens = torch.cat((curr_step_batch_one_hot[:,0,:],curr_step_batch_one_hot[:,1,:]), dim=-1)
        else:
            new_tens = curr_step_batch_one_hot[0]
            if not one_at_a_time:
                range_end *= self.history_len

            for i in range(1, range_end):
                new_tens = torch.cat((new_tens, curr_step_batch_one_hot[i]), dim=-1)

        curr_step_batch = new_tens.float()
        return curr_step_batch

    def rollout(self, th, vals):

        init_state_batch = self.get_init_state_batch()

        state_batch = init_state_batch

        if self.state_type == 'mnist':
            obs_history = torch.zeros((self.num_iters, self.batch_size, self.n_agents * self.history_len, 28, 28))
        elif self.state_type == 'majorTD4':
            obs_history = torch.zeros((self.num_iters, self.n_agents, self.batch_size, 2 * self.action_repr_dim * self.history_len))
        else:
            obs_history = torch.zeros((self.num_iters, self.batch_size, self.n_agents * self.action_repr_dim * self.history_len))
        # trajectory just tracks actions, doesn't track the init state
        action_trajectory = torch.zeros((self.num_iters, self.n_agents, self.batch_size, 1), dtype=torch.int)
        rewards = torch.zeros((self.num_iters, self.n_agents, self.batch_size, 1))
        policy_history = torch.zeros((self.num_iters, self.n_agents, self.batch_size, 1))
        val_history = torch.zeros((self.num_iters, self.n_agents, self.batch_size, 1))

        # This loop can't be skipped due to sequential nature of environment
        for iter in range(self.num_iters):

            policies, state_values = self.get_policy_vals_indices_for_iter(th, vals, state_batch, iter)

            policy_history[iter] = policies
            val_history[iter] = state_values

            actions = torch.distributions.binomial.Binomial(probs=policies.detach()).sample()

            curr_step_batch = actions

            total_contrib = sum(actions)

            if self.state_type == 'one_hot':
                curr_step_batch = self.build_one_hot_from_batch(curr_step_batch, self.action_repr_dim)
            elif self.state_type == 'majorTD4':
                curr_step_batch = torch.zeros((self.n_agents, self.batch_size, 2 * self.action_repr_dim * self.history_len))
                for i in range(self.n_agents):
                    num_other_contributors = total_contrib - actions[i]

                    majority_coop = (num_other_contributors / (self.n_agents - 1.)) >= 0.5
                    individual_obs = torch.cat((actions[i], majority_coop), dim=-1)

                    curr_step_batch[i] = self.build_one_hot_from_batch(individual_obs,
                                                  self.action_repr_dim, simple_2state_build=True)

            else:
            # This awkward reshape and transpose stuff gets around some issues with reshaping not preserving the data in the ways I want
                curr_step_batch = curr_step_batch.reshape(self.n_agents, self.batch_size)
                curr_step_batch = curr_step_batch.t()

            if self.history_len > 1:
                if self.state_type == 'majorTD4':
                    raise NotImplementedError("Probably needs extra dimension at start for below stuff")

                new_state_batch = torch.zeros_like(state_batch)
                new_state_batch[:, :self.n_agents * self.action_repr_dim * (self.history_len-1)] = state_batch[:, self.n_agents * self.action_repr_dim :self.n_agents * self.action_repr_dim  * self.history_len]
                new_state_batch[:, self.n_agents * self.action_repr_dim * (self.history_len - 1):] = curr_step_batch

                state_batch = new_state_batch
            else:
                state_batch = curr_step_batch

            if self.state_type == 'mnist':
                state_batch = self.build_mnist_state_from_classes(state_batch)

            action_trajectory[iter] = torch.Tensor(actions)

            obs_history[iter] = state_batch

            payout_per_agent = total_contrib * self.contribution_factor / self.n_agents
            agent_rewards = -actions + payout_per_agent  # if agent contributed 1, subtract 1, that's what the -actions does
            # Note negative rewards might help with exploration in PG formulation
            # (e.g. if you shift all rewards downward by the maximum possible value, so that all rewards are negative),
            # when you have a policy gradient formulation without value function (or even with value function at the beginning of training when values are close to 0)
            # this will encourage exploration (trying actions that haven't been done before)
            # But with advantage/actor-critic formulation, shouldn't matter much
            agent_rewards -= adjustment_to_make_rewards_negative
            rewards[iter] = agent_rewards

        next_val_history = self.get_next_val_history(th, vals, val_history, state_batch, iter + 1) # iter doesn't even matter here as long as > 0

        return action_trajectory, rewards, policy_history, val_history, next_val_history, obs_history

    def get_loss_helper(self, trajectory, rewards, policy_history, old_policy_history = None):
        num_iters = len(trajectory)

        discounts = torch.cumprod(self.gamma * torch.ones((num_iters)),
                                  dim=0) / self.gamma

        # Discounted rewards, not sum of rewards which G_t is
        gamma_t_r_ts = rewards * discounts.reshape(-1, 1, 1, 1)  # implicit broadcasting done by numpy

        G_ts = reverse_cumsum(gamma_t_r_ts  , dim=0)
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

            return G_ts, gamma_t_r_ts, p_act_ratio, discounts

    def get_gradient_terms(self, trajectory, rewards, policy_history):

        G_ts, gamma_t_r_ts, log_p_act = self.get_loss_helper(trajectory,
                                                             rewards,
                                                             policy_history)

        # These are basically grad_i E[R_0^i] - naive learning loss
        # no LOLA loss here yet
        objective_nl = (log_p_act * G_ts).sum(dim=0)

        log_p_times_G_t_matrix = torch.zeros((self.n_agents, self.n_agents))
        # so entry 0,0 is - (log_p_act[:,0] * G_ts[:,0]).sum(dim=0)
        # entry 1,1 is - (log_p_act[:,1] * G_ts[:,1]).sum(dim=0)
        # and so on
        # entry 0,1 is - (log_p_act[:,0] * G_ts[:,1]).sum(dim=0)
        # and so on
        # Be careful with dimensions/not to mix them up

        for i in range(self.n_agents):
            for j in range(self.n_agents):
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

        # NOTE THESE ARE NOT LOSSES, THEY ARE REWARDS (discounted)
        # Need to negative if you will torch optim on them. BE CAREFUL WITH THIS

        grad_log_p_act = []

        for i in range(self.n_agents):
            # Could probably get this without taking grad, could be more efficient

            # CANNOT DO mean here. Must do individually for every batch, preserving the batch_size dimension
            # until later.

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

    def build_policy_dist(self, coop_prob_history_all_agents, i):
        coop_prob_i = coop_prob_history_all_agents[:, i, :]
        defect_prob_i = 1 - coop_prob_i
        policy_dist_i = torch.cat((coop_prob_i, defect_prob_i),
                                  dim=-1)  # we need to do this because kl_div needs the full distribution
        # and the way we have parameterized policy here is just a coop prob
        # if you used categorical/multinomial you wouldn't have to go through this
        # so maybe I should replace as categorical?
        policy_dist_i = policy_dist_i.reshape(self.batch_size, self.num_iters,
                                              -1)
        return policy_dist_i

    def get_dice_loss(self, trajectory, rewards, policy_history, val_history, next_val_history,
                      old_policy_history=None, kl_div_target_policy=None, use_nl_loss=False, use_clipping=False, use_penalty=False, beta=None):

        if old_policy_history is not None:
            old_policy_history = old_policy_history.detach()

        G_ts, gamma_t_r_ts, log_p_act_or_p_act_ratio, discounts = self.get_loss_helper(
            trajectory, rewards, policy_history, old_policy_history)

        discounts = discounts.view(-1, 1, 1, 1)

        # R_t is like G_t except not discounted back to the start. It is the forward
        # looking return at that point in time
        R_ts = G_ts / discounts

        # Generalized Advantage Estimation (GAE) calc adapted from loaded dice repo
        # https://github.com/oxwhirl/loaded-dice/blob/master/loaded_dice_demo.ipynb
        advantages = torch.zeros_like(G_ts)
        lambd = 0 #0.95 # 1 here is essentially what I was doing before with monte carlo
        deltas = rewards + self.gamma * next_val_history.detach() - val_history.detach()
        gae = torch.zeros_like(deltas[0,:]).float()
        for i in range(deltas.size(0) - 1, -1, -1):
            gae = gae * gamma * lambd + deltas[i,:]
            advantages[i,:] = gae

        if repeat_train_on_same_samples:
            # Then we should have a p_act_ratio here instead of a log_p_act
            if use_clipping:

                # Two way clamp, not yet ppo style
                if two_way_clip:
                    log_p_act_or_p_act_ratio = torch.clamp(log_p_act_or_p_act_ratio, min=1 - clip_epsilon, max=1 + clip_epsilon)
                else:
                    # PPO style clipping
                    pos_adv = (advantages > 0).float()
                    log_p_act_or_p_act_ratio = pos_adv * torch.minimum(log_p_act_or_p_act_ratio,torch.zeros_like(log_p_act_or_p_act_ratio) + 1+clip_epsilon) + \
                                               (1-pos_adv) * torch.maximum(log_p_act_or_p_act_ratio,torch.zeros_like(log_p_act_or_p_act_ratio) + 1-clip_epsilon)

            if use_penalty:
                # Calculate KL Divergence
                kl_divs = torch.zeros((self.n_agents))

                if kl_div_target_policy is None:
                    assert old_policy_history is not None
                    kl_div_target_policy = old_policy_history

                for i in range(self.n_agents):

                    policy_dist_i = self.build_policy_dist(policy_history, i)
                    kl_target_dist_i = self.build_policy_dist(kl_div_target_policy, i)

                    kl_div = torch.nn.functional.kl_div(input=torch.log(policy_dist_i),
                                                    target=kl_target_dist_i.detach(),
                                                    reduction='batchmean',
                                                    log_target=False)
                    # print(kl_div)
                    kl_divs[i] = kl_div

                # print(kl_divs)

        sum_over_agents_log_p_act_or_p_act_ratio = log_p_act_or_p_act_ratio.sum(dim=1)

        # See 5.2 (page 7) of DiCE paper for below:
        # With batches, the mean is the mean across batches. The sum is over the steps in the rollout/trajectory

        deps_up_to_t = (torch.cumsum(sum_over_agents_log_p_act_or_p_act_ratio, dim=0)).reshape(-1, 1, self.batch_size, 1)

        deps_less_than_t = deps_up_to_t - sum_over_agents_log_p_act_or_p_act_ratio.reshape(-1, 1, self.batch_size, 1) # take out the dependency in the given time step

        # Look at Loaded DiCE paper to see where this formulation comes from
        # Right now since I am using GAE, the advantages already have the discounts in them, no need to multiply again
        loaded_dice_rewards = ((magic_box(deps_up_to_t) - magic_box(deps_less_than_t)) * advantages).sum(dim=0).mean(dim=1)

        dice_loss = -loaded_dice_rewards

        if repeat_train_on_same_samples and use_penalty:
            kl_divs = kl_divs.unsqueeze(-1)

            assert beta is not None

            # TODO make adaptive
            dice_loss += beta * kl_divs # we want to min the positive kl_div

        final_state_vals = next_val_history[-1].detach()
        # You DO need a detach on these. Because it's the target - it should be detached. It's a target value.
        values_loss = ((R_ts + (self.gamma * discounts.flip(dims=[0])) * final_state_vals.reshape(1, *final_state_vals.shape) - val_history) ** 2).sum(dim=0).mean(dim=1)

        if use_nl_loss:
            # No LOLA/opponent shaping or whatever, just naive learning
            regular_nl_loss = -(log_p_act_or_p_act_ratio * advantages).sum(dim=0).mean(dim=1)
            # Well I mean obviously if you do this there is no shaping because you can't differentiate through the inner update step...
            return regular_nl_loss, G_ts, values_loss


        return dice_loss, G_ts, values_loss



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
        init = torch.rand(dims[i], requires_grad=True)
        th.append(init)
    print("Policies:")
    print(th)
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


def init_th_adversarial(dims):
    th = []
    for i in range(len(dims)):
        # For some reason this -0 is needed
        init = torch.zeros(dims[i], requires_grad=True) - 0
        init[0] -= 5
        th.append(init)
    return th

def init_th_adversarial2(dims):
    th = []
    for i in range(len(dims)):
        # For some reason this -0 is needed
        init = torch.zeros(dims[i], requires_grad=True) - 2
        th.append(init)
    return th

def init_th_adversarial3(dims):
    th = []
    for i in range(len(dims)):
        # For some reason this -0 is needed
        init = torch.zeros(dims[i], requires_grad=True) - 0
        th.append(init)
    th[0][3] += 0.5
    th[1][2] += 0.5
    return th

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, extra_hidden_layers,
                 output_size, final_sigmoid=False, final_softmax=False):

        # Dec 28 2021 reformulated to not use final_sigmoid, so that
        # I can play around with conditioning matrices on the logits
        super(NeuralNet, self).__init__()
        layers = []

        layers.append(torch.nn.Linear(input_size, hidden_size))
        # layers.append(torch.nn.LeakyReLU(negative_slope=0.01)) # swap in if you want. Should be made into a dynamic argument
        layers.append(torch.nn.Tanh())
        for i in range(extra_hidden_layers):
            layers.append(nn.Linear(hidden_size, hidden_size))
            # layers.append(torch.nn.LeakyReLU(negative_slope=0.01))
            layers.append(torch.nn.Tanh())
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
    def __init__(self, conv_in_channels, conv_out_channels, input_size, hidden_size, output_size, kernel_size=5, final_sigmoid=False):
        super(ConvFC, self).__init__()

        self.conv_out_channels = conv_out_channels

        self.layer1 = nn.Conv2d(conv_in_channels, conv_out_channels,
                                kernel_size=kernel_size)
        self.conv_result_size = (input_size - kernel_size + 1)  # no stride or pad here
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


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_rnn_layers, final_softmax):
        super().__init__()
        self.RNN = nn.RNN(input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=num_rnn_layers,
                          nonlinearity='tanh',
                          batch_first=True)
        self.linear = nn.Linear(hidden_size, 4)
        self.final_softmax = final_softmax
        self.init_hidden_state = torch.zeros([num_rnn_layers, args.batch_size, hidden_size]).requires_grad_(True)

    def forward(self, x):
        output, hn = self.RNN(x)
        out = self.linear(output[:, -1, :])
        if self.final_softmax:
            out = torch.nn.functional.softmax(out, dim=-1)
        return out


# TODO maybe this should go into the game definition itself and be part of that class instead of separate
def init_custom(dims, state_type, using_nn=True, using_rnn=False, env='ipd', nn_hidden_size=16, nn_extra_hidden_layers=0):
    th = []
    # f_th = []

    # NN/func approx
    if using_nn:
        for i in range(len(dims)):

            if state_type == 'mnist':
                assert env == 'ipd'

                # dims[i] here is the number of agents. Because there is one MNIST
                # digit per agent's last action, thus we have an n-dimensional
                # set of MNIST images
                # conv out channels could be something other than dims[i] if you wanted.
                policy_net = ConvFC(conv_in_channels=dims[i],
                                    # mnist specific input is 28x28x1
                                    conv_out_channels=dims[i],
                                    input_size=28,
                                    hidden_size=nn_hidden_size,
                                    output_size=1,
                                    final_sigmoid=False)
            else:

                if env == 'coin':
                    if using_rnn:
                        policy_net = RNN(input_size=dims[i], hidden_size=nn_hidden_size,
                                         num_rnn_layers=nn_extra_hidden_layers+1, final_softmax=True) # 1 rnn layer
                    else:
                        policy_net = NeuralNet(input_size=dims[i],
                                           hidden_size=nn_hidden_size,
                                           extra_hidden_layers=nn_extra_hidden_layers,
                                           output_size=4, final_sigmoid=False, final_softmax=True) # TODO probably should dynamically code this
                else:
                    if env == 'hawkdove':
                        policy_net = NeuralNet(input_size=dims[i],
                                               hidden_size=nn_hidden_size,
                                               extra_hidden_layers=nn_extra_hidden_layers,
                                               output_size=n_agents)
                    else:
                        policy_net = NeuralNet(input_size=dims[i], hidden_size=nn_hidden_size, extra_hidden_layers=nn_extra_hidden_layers,
                                  output_size=1)

            # f_policy_net = higher.patch.monkeypatch(policy_net, copy_initial_weights=True,
            #                          track_higher_grads=False)

            # print(f_policy_net)

            th.append(policy_net)

            # f_th.append(f_policy_net)

            # th.append(f_policy_net)

    # Tabular policies
    else:
        for i in range(len(dims)):
            # DONT FORGET THIS +1
            # Right now if you omit the +1 we get a bug where the first state is the prob in the all contrib state
            th.append(torch.nn.init.normal_(torch.empty(2**n_agents + 1, requires_grad=True), std=args.std))


    # optims_th = construct_optims(th, lr_policies)
    # optims = None

    assert len(th) == len(dims)
    # assert len(optims) == len(dims)

    vals = []
    # f_vals = []

    for i in range(len(dims)):
        if using_nn:
            if state_type == 'mnist':
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
            # f_vals_net = higher.patch.monkeypatch(vals_net,
            #                                         copy_initial_weights=True,
            #                                         track_higher_grads=False)
            # print(f_policy_net)

            # f_vals.append(f_vals_net)
        else:
            vals.append(torch.nn.init.normal_(torch.empty(2**n_agents + 1, requires_grad=True), std=args.std))

    assert len(vals) == len(dims)

    optims_vals = construct_optims(vals, lr_values)

    # diff_optims_th = construct_diff_optims(th, lr_policies, f_th)
    # diff_optims_vals = construct_diff_optims(vals, lr_values, f_vals)

    return th, vals, optims_vals #, f_th, f_vals #, diff_optims_th, diff_optims_vals

def construct_diff_optims(th_or_vals, lrs, f_th_or_vals):
    optims = []
    for i in range(len(th_or_vals)):
        if not isinstance(th_or_vals[i], torch.Tensor):
            optim = torch.optim.SGD(th_or_vals[i].parameters(), lr=lrs[i])
            diffoptim = higher.get_diff_optim(optim, th_or_vals[i].parameters(), f_th_or_vals[i])
            optims.append(diffoptim)
        else:
            # Don't use for now, not tested
            print("Warning: be careful here")
            optim = torch.optim.SGD([th_or_vals[i]], lr=lrs[i])
            diffoptim = higher.optim.DifferentiableSGD(optim, [th_or_vals[i]])
            optims.append(diffoptim)
    return optims

def construct_optims(th_or_vals, lrs):
    optims = []
    for i in range(len(th_or_vals)):
        if not isinstance(th_or_vals[i], torch.Tensor):
            optim = torch.optim.SGD(th_or_vals[i].parameters(), lr=lrs[i])
            optims.append(optim)
        else:
            # Don't use for now, not tested
            print("Warning: be careful here")
            optim = torch.optim.SGD([th_or_vals[i]], lr=lrs[i])
            optims.append(optim)
    return optims

def get_gradient(function, param):
    grad = torch.autograd.grad(function, param, create_graph=True)[0]
    return grad

def get_jacobian(terms, param):
    jac = []
    for term in terms:
        grad = torch.autograd.grad(term, param, retain_graph=True, create_graph=False)[0]
        jac.append(grad)
    jac = torch.vstack(jac)
    return jac


def get_th_copy(th):
    static_th_copy = []
    for i in range(len(th)):
        if isinstance(th[i], torch.Tensor):
            static_th_copy.append(th[i].detach().clone().requires_grad_())
        else:
            raise NotImplementedError(
                "To be implemented, use copyNN function (need reconstruction of NNs?)")
    return static_th_copy


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

def build_policy_and_target_policy_dists(policy_to_build, target_pol_to_build, i):
    # Note the policy and targets are individual agent ones
    if args.ill_condition:
        policy_dist = build_policy_dist(
            torch.sigmoid(ill_cond_matrices[i] @ policy_to_build))

        target_policy_dist = build_policy_dist(
            torch.sigmoid(ill_cond_matrices[i] @ target_pol_to_build.detach()))
    else:
        policy_dist = build_policy_dist(torch.sigmoid(policy_to_build))
        target_policy_dist = build_policy_dist(
            torch.sigmoid(target_pol_to_build.detach()))
    return policy_dist, target_policy_dist


# TODO perhaps the inner_loop step can pass the lr to this prox_f and we can scale the inner lr by eta, if we want more control over it

def prox_f(th_to_build_on, kl_div_target_th, Ls, j, prox_f_step_sizes, iters = 0, max_iters = 1000, threshold = 1e-8):
    # For each other player, do the prox operator
    # (this function just does on a single player, it should be used within the loop iterating over all players)
    # We will do this by gradient descent on the proximal objective
    # Until we reach a fixed point, which tells use we have reached
    # the minimum of the prox objective, which is our prox operator result
    fixed_point_reached = False

    curr_pol = th_to_build_on[j].detach().clone()

    while not fixed_point_reached:

        inner_rews = Ls(th_to_build_on)

        policy_dist, target_policy_dist = build_policy_and_target_policy_dists(th_to_build_on[j], kl_div_target_th[j], j)

        kl_div = torch.nn.functional.kl_div(input=torch.log(policy_dist),
                                            target=target_policy_dist,
                                            reduction='batchmean',
                                            log_target=False)
        # Again we have this awkward reward formulation
        loss_j = - inner_rews[j] + args.inner_beta * kl_div
        # No eta here because we are going to solve it exactly anyway

        with torch.no_grad():
            th_to_build_on[j] -= prox_f_step_sizes[j] * get_gradient(loss_j,
                                                               th_to_build_on[
                                                                       j])

        prev_pol = curr_pol.detach().clone()
        curr_pol = th_to_build_on[j].detach().clone()

        iters += 1

        policy_dist, target_policy_dist = build_policy_and_target_policy_dists(curr_pol, prev_pol, j)

        curr_prev_pol_div = torch.nn.functional.kl_div(
            input=torch.log(policy_dist),
            target=target_policy_dist,
            reduction='batchmean',
            log_target=False)

        if curr_prev_pol_div < threshold or iters > max_iters:
            if args.print_prox_loops_info:
                print("Inner prox iters used: {}".format(iters))
            if iters >= max_iters:
                print("Reached max prox iters")
            fixed_point_reached = True

    return th_to_build_on[j].detach().clone().requires_grad_()


def get_ift_terms(inner_lookahead_th, kl_div_target_th, gradient_terms_or_Ls, i, j):
    rews_for_ift = gradient_terms_or_Ls(inner_lookahead_th)  # Note that new_th has only agent j updated

    grad2_V1 = get_gradient(- rews_for_ift[i], inner_lookahead_th[j])

    # We use inner_lookahead_th instead of new_th because inner_lookahead has only th[j] updated
    policy_dist, target_policy_dist = build_policy_and_target_policy_dists(inner_lookahead_th[j], kl_div_target_th[j], j)

    kl_div = torch.nn.functional.kl_div(
        input=torch.log(policy_dist),
        target=target_policy_dist,
        reduction='batchmean',
        log_target=False)

    # Again we have this awkward reward formulation
    loss_j = - rews_for_ift[j] + args.inner_beta * kl_div
    # Right the LR here shouldn't matter because it's a fixed point so the gradient should be close to 0 anyway.
    # BUT it will make an effect on the outer gradient update
    f_at_fixed_point = inner_lookahead_th[j] - lr_policies_inner[j] * get_gradient(
        loss_j, inner_lookahead_th[j])

    print_info = args.print_prox_loops_info
    if print_info:
        print(inner_lookahead_th[j])

    grad0_f = get_jacobian(f_at_fixed_point, inner_lookahead_th[i])
    grad1_f = get_jacobian(f_at_fixed_point, inner_lookahead_th[j])

    # This inverse can fail when init_state_rep = 1
    mat_to_inv = torch.eye(th[j].shape[0]) - grad1_f
    if mat_to_inv[-1, -1] == 0:
        print("UHOH - something bad is happening")
        mat_to_inv[
            -1, -1] += 1e-6  # for numerical stability/to prevent inverse failing
        # Something bad has happened if you got to this point here though.
        # Try changing hyperparameters to avoid instability, looking through other issues with code or settings

    grad_th1_th2prime = torch.inverse(mat_to_inv) @ grad0_f

    return grad2_V1, grad_th1_th2prime


def inner_exact_loop_step(starting_th, kl_div_target_th, gradient_terms_or_Ls, i, n, prox_f_step_sizes):
    other_terms = []
    new_th = get_th_copy(starting_th)

    # i is the agent doing the rolling out of the other agents
    for j in range(n):
        if j != i:
            # Inner lookahead th has only agent j's th being updated
            inner_lookahead_th = get_th_copy(starting_th)

            # Inner loop essentially
            # Each player on the copied th does a naive update (doesn't have to be differentiable here because of fixed point/IFT)
            if not isinstance(inner_lookahead_th[j], torch.Tensor):
                raise NotImplementedError
            else:
                # For each other player, do the prox operator
                inner_lookahead_th[j] = prox_f(inner_lookahead_th, kl_div_target_th,
                                               gradient_terms_or_Ls, j, prox_f_step_sizes)
                # new_th[j] = inner_lookahead_th[j].detach().clone().requires_grad_()
                # You could do this without the detach, and using x = x - grad instead of -= above, and no torch.no_grad
                # And then differentiate through the entire process
                # But we want instead fixed point / IFT for efficiency and not having to differentiate through entire unroll process.
                # For nice IFT primer, check out https://implicit-layers-tutorial.org/implicit_functions/

            grad2_V1, grad_th1_th2prime = get_ift_terms(inner_lookahead_th,
                                                        kl_div_target_th,
                                                        gradient_terms_or_Ls, i, j)

            other_terms.append(grad2_V1 @ grad_th1_th2prime)

            new_th[j] = inner_lookahead_th[j]

    return new_th, other_terms


def outer_exact_loop_step(print_info, i, new_th, static_th_copy, Ls, curr_pol, other_terms, curr_iter, max_iters, threshold):
    if print_info:
        for j in range(len(new_th)):
            if j != i:
                print("Agent {}".format(j))
                print(torch.sigmoid(new_th[j]))

    outer_rews = Ls(new_th)

    policy_dist, target_policy_dist = build_policy_and_target_policy_dists(
        new_th[i], static_th_copy[i], i)

    kl_div = torch.nn.functional.kl_div(
        input=torch.log(policy_dist),
        target=target_policy_dist,
        reduction='batchmean',
        log_target=False)

    # Again we have this awkward reward formulation
    loss_i = - outer_rews[i] + args.outer_beta * kl_div

    with torch.no_grad():
        if other_terms is not None:
            new_th[i] -= lr_policies_outer[i] * (
                        get_gradient(loss_i, new_th[i]) + sum(other_terms))
        else:
            new_th[i] -= lr_policies_outer[i] * (get_gradient(loss_i, new_th[i]))

    prev_pol = curr_pol.detach().clone()
    curr_pol = new_th[i].detach().clone()

    if print_info:
        print(kl_div)
        print("Curr pol")
        print(torch.sigmoid(curr_pol))
        print("Iter:")
        print(curr_iter)

    policy_dist, target_policy_dist = build_policy_and_target_policy_dists(
        curr_pol, prev_pol, i)

    curr_prev_pol_div = torch.nn.functional.kl_div(
        input=torch.log(policy_dist),
        target=target_policy_dist,
        reduction='batchmean',
        log_target=False)

    fixed_point_reached = False
    if curr_prev_pol_div < threshold or curr_iter > max_iters:
        print("Outer prox iters used: {}".format(curr_iter))
        if curr_iter >= max_iters:
            print("Reached max prox iters")
        fixed_point_reached = True

    return new_th, curr_pol, fixed_point_reached


def print_exact_policy(th, i):
    # Used for exact gradient setting
    print(
        "---Agent {} Rollout---".format(i + 1))
    for j in range(len(th)):
        print("Agent {} Policy".format(j+1))
        print(torch.sigmoid(th[j]))

        if args.ill_condition:
            print("Agent {} Transformed Policy".format(j + 1))
            print(torch.sigmoid(ill_cond_matrices[j] @ th[j]))


def exact_grad_calc(th, gradient_terms_or_Ls):
    if args.no_taylor_approx:
        static_th_copy = get_th_copy(th)
        n = len(th)
        losses = gradient_terms_or_Ls(th)

        grad_L = [[get_gradient(losses[j], th[i]) for j in range(n)] for i in
                  range(n)]

        nl_grads = [grad_L[i][i] for i in range(n)]
        lola_grads = []

        for i in range(n):
            new_th = get_th_copy(static_th_copy)
            # Then each player calcs the losses
            inner_losses = gradient_terms_or_Ls(new_th)

            for j in range(n):
                # Inner loop essentially
                # Each player on the copied th does a naive update (must be differentiable!)
                if j != i:
                    if not isinstance(new_th[j], torch.Tensor):
                        raise NotImplementedError
                    else:
                        new_th[j] = new_th[j] + lr_policies_inner[
                            j]  * get_gradient(inner_losses[j],
                                                    new_th[j])

            # Then each player recalcs losses using mixed th where everyone else's is the new th but own th is the old (copied) one (do this in a for loop)
            outer_losses = gradient_terms_or_Ls(new_th)

            # Finally calc grad
            lola_grad = get_gradient(outer_losses[i], new_th[i])
            lola_grads.append(lola_grad)

            # Here the goal is to compare LOLA gradient with the naive gradient
            # Naive would be no inner lookahead
            # And then the difference between those tells us
            # in what direction the lola shaping term is

        print("!!!NL TERMS!!!")
        print(nl_grads[0])
        print(nl_grads[1])

        lola_terms_p1 = lola_grads[0] - nl_grads[0]
        lola_terms_p2 = lola_grads[1] - nl_grads[1]
        print("!!!LOLA TERMS!!!")
        print(lola_terms_p1)
        print(lola_terms_p2)

    else:
        n = len(th)
        losses = gradient_terms_or_Ls(th)

        grad_L = [[get_gradient(losses[j], th[i]) for j in range(n)] for i in
                  range(n)]

        terms = [sum([torch.dot(grad_L[j][i], grad_L[j][j])
                      for j in range(n) if j != i]) for i in range(n)]

        lola_terms = [
            lr_policies_inner[i] * get_gradient(terms[i], th[i])
            for i in range(n)]

        nl_terms = [grad_L[i][i]
                    for i in range(n)]

        print("!!!NL TERMS!!!")
        print(nl_terms)
        print("!!!LOLA TERMS!!!")
        print(lola_terms)

        assert n == 2 # not yet supporting more agents

        lola_terms_p1 = lola_terms[0]
        lola_terms_p2 = lola_terms[1]

    is_in_tft_direction_p1, is_in_tft_direction_p2 = check_is_in_tft_direction(lola_terms_p1, lola_terms_p2)

    return is_in_tft_direction_p1, is_in_tft_direction_p2


def check_is_in_tft_direction(lola_terms_p1, lola_terms_p2):
    is_in_tft_direction_p1 = lola_terms_p1[0] < 0 and \
                             lola_terms_p1[1] > 0 and \
                             lola_terms_p1[2] < 0 and \
                             lola_terms_p1[3] > 0
    is_in_tft_direction_p2 = lola_terms_p2[0] < 0 and \
                             lola_terms_p2[1] < 0 and \
                             lola_terms_p2[2] > 0 and \
                             lola_terms_p2[3] > 0
    print("P1 LOLA TFT Direction? {}".format(
        is_in_tft_direction_p1))
    print("P2 LOLA TFT Direction? {}".format(
        is_in_tft_direction_p2))
    return is_in_tft_direction_p1, is_in_tft_direction_p2


def update_th(th, gradient_terms_or_Ls, lr_policies_outer, lr_policies_inner, algos, using_samples):
    # Sorry about gradient_terms_or_Ls. Kind of an inelegant solution to shove both in the same variable
    # The vast majority of the time, gradient_terms_or_Ls is just Ls, and should be thought of that way
    # Recall Ls is the function built from the ipdn function

    n = len(th)

    G_ts = None
    grad_2_return_1 = None
    nl_terms = None

    if using_samples:
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

    # Be careful with mixed algorithms here; I have not tested it much
    if 'lola' in algos:

        if using_samples:
            # This section here is essentially LOLA-PG in the original paper
            # This is not using DiCE
            # It is quite ugly for neural nets - not recommended to use this formulation
            # for neural nets. Use DiCE instead.

            grad_1_grad_2_return_2_new = []
            for i in range(n_agents):
                grad_1_grad_2_return_2_new.append([0] * n_agents)


            grad_log_p_act_sums_0_to_t = []
            for i in range(n_agents):
                grad_log_p_act_sums_0_to_t.append(torch.cumsum(grad_log_p_act[i], dim=0))

            for i in range(n_agents):
                for j in range(n_agents):
                    if i != j:
                        for t in range(rollout_len):
                            for b in range(batch_size):

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

            # Eta is a hyperparam (eta = lr_policies_inner for the particular agent; eta was used in LOLA paper)

            # IMPORTANT NOTE: the way these grad return matrices are set up is that you should always call i j here
            # because the j i switch was done during the construction of the matrix
            lola_terms = [sum([lr_policies_inner[i] * grad_2_return_1[i][j].t() @
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
            print_info = args.print_prox_loops_info

            if args.inner_exact_prox:
                static_th_copy = get_th_copy(th)

                for i in range(n):
                    if args.outer_exact_prox:

                        new_th = get_th_copy(static_th_copy)
                        # new_th will hold the original player i theta, but for all other players j,
                        # their new theta after they do their INDIVIDUAL inner loop update in which
                        # they only update their own theta (and not all the other players) (which is really player i imagining the other players individually doing updates)

                        # Here with outer exact prox and inner exact prox,
                        # we have a loop where we do: inner_exact_loop_step, then take one outer grad step on proximal objective, repeat
                        # until diff between outer loop policies is small so outer loop has reached some fixed point.
                        # With the exact proximal algorithm using the IFT for differentiation
                        # we do not care about the path from start to fixed point
                        # We only care about the final fixed point
                        # Therefore for subsequent iterations, to reduce the time it takes for us to get to the fixed point,
                        # we warm start from the previous iteration's fixed point instead of starting from the beginning again
                        # Conversely, as can be seen in the approximate version of this algorithm, when taking limited steps, we
                        # cannot do the same warm start; since we are unrolling through the limited steps, and not using IFT, the path
                        # we take to get to the new point matters.

                        fixed_point_reached = False
                        outer_iters = 0

                        curr_pol = new_th[i].detach().clone()
                        while not fixed_point_reached:
                            if print_info:
                                print("loop start")
                                for j in range(len(new_th)):
                                    if j != i:
                                        print("Agent {}".format(j))
                                        print(torch.sigmoid(new_th[j]))

                            new_th, other_terms = inner_exact_loop_step(new_th, static_th_copy, gradient_terms_or_Ls, i, n,
                                                                        prox_f_step_sizes=lr_policies_inner)

                            outer_iters += 1
                            new_th, curr_pol, fixed_point_reached = outer_exact_loop_step(print_info, i, new_th, static_th_copy, gradient_terms_or_Ls, curr_pol,
                                                                                other_terms, outer_iters, args.prox_max_iters, args.prox_threshold)

                        th[i] = new_th[i]

                    else:
                        # No outer exact prox
                        # This is just a single gradient step on the outer step:
                        # That is, we calc the inner loop exactly
                        # Use IFT to differentiate through and then get the outer gradient
                        # Take 1 step, and then that's it. Move on to next loop/iteration/agent

                        new_th, other_terms = inner_exact_loop_step(static_th_copy, static_th_copy, gradient_terms_or_Ls, i, n,
                                                                    prox_f_step_sizes=lr_policies_inner)

                        outer_rews = gradient_terms_or_Ls(new_th)

                        nl_grad = get_gradient(-outer_rews[i], new_th[i])

                        with torch.no_grad():
                            new_th[i] -= lr_policies_outer[i] * (nl_grad + sum(other_terms))

                        th[i] = new_th[i]
                return th, losses, G_ts, nl_terms, None, grad_2_return_1


            elif args.no_taylor_approx:
                # Do DiCE style rollouts except we can calculate exact Ls like follows

                # So what we will do is each player will calc losses
                # First copy the th
                static_th_copy = get_th_copy(th)

                for i in range(n):
                    new_th = get_th_copy(static_th_copy)

                    if args.outer_exact_prox:

                        fixed_point_reached = False
                        outer_iters = 0

                        curr_pol = new_th[i].detach().clone()
                        inner_losses = gradient_terms_or_Ls(new_th)

                        for j in range(n):
                            # Inner loop essentially
                            # Each player on the copied th does a naive update (must be differentiable!)
                            if j != i:
                                if not isinstance(new_th[j], torch.Tensor):
                                    raise NotImplementedError
                                else:
                                    new_th[j] = new_th[j] + lr_policies_inner[
                                        j] * get_gradient(
                                        inner_losses[j], new_th[j])

                        if args.print_inner_rollouts:
                            print_exact_policy(new_th, i)

                        while not fixed_point_reached:
                            # Then each player calcs the losses
                            outer_iters += 1

                            new_th, curr_pol, fixed_point_reached = outer_exact_loop_step(print_info, i, new_th, static_th_copy,
                                                                                gradient_terms_or_Ls, curr_pol,
                                                                                None, outer_iters, args.prox_max_iters, args.prox_threshold)
                        th[i] = new_th[i]

                    else:
                        # Then each player calcs the losses
                        inner_losses = gradient_terms_or_Ls(new_th)

                        for j in range(n):
                            # Inner loop essentially
                            # Each player on the copied th does a naive update (must be differentiable!)
                            if j != i:
                                if not isinstance(new_th[j], torch.Tensor):
                                    raise NotImplementedError
                                else:
                                    new_th[j] = new_th[j] + lr_policies_inner[
                                        j] * get_gradient(inner_losses[j],
                                                                new_th[j])


                        if args.print_inner_rollouts:
                            print_exact_policy(new_th, i)

                        # Then each player recalcs losses using mixed th where everyone else's is the new th but own th is the old (copied) one (do this in a for loop)
                        outer_losses = gradient_terms_or_Ls(new_th)

                        # Finally each player updates their own (copied) th
                        with torch.no_grad():
                            new_th[i] += lr_policies_outer[i] * get_gradient(outer_losses[i], new_th[i])

                        # Finally we rewrite the th by copying from the created copies
                        th[i] = new_th[i]

                return th, losses, G_ts, nl_terms, None, grad_2_return_1

            else:
                # Here we continue with the exact gradient calculations
                # We are using the SOS formulation that doesn't drop the term that
                # the original LOLA dropped
                # Look at pg 12, Stable Opponent Shaping paper
                # This is the first line of the second set of equations
                # sum over all j != i of grad_j L_i * grad_j L_j
                # Then that's your lola term once you differentiate it with respect to theta_i
                # Then add the naive learning term
                terms = [sum([torch.dot(grad_L[j][i], grad_L[j][j])
                              for j in range(n) if j != i]) for i in range(n)]

                lola_terms = [
                    lr_policies_inner[i] * get_gradient(terms[i], th[i])
                    for i in range(n)]


                nl_terms = [grad_L[i][i]
                            for i in range(n)]

                print("!!!NL TERMS!!!")
                print(nl_terms)
                if args.ill_condition:
                    for i in range(n):
                        print(ill_cond_matrices[i] @ nl_terms[i])
                print("!!!LOLA TERMS!!!")
                print(lola_terms)
                if args.ill_condition:
                    for i in range(n):
                        print(ill_cond_matrices[i] @ lola_terms[i])

                grads = [nl_terms[i] + lola_terms[i] for i in range(n)]

    else:  # Naive Learning, no LOLA/opponent shaping
        grads = [grad_L[i][i] for i in range(n)]

    # Update theta
    with torch.no_grad():
        for i in range(n):
            if not isinstance(th[i], torch.Tensor):
                k = 0
                for param in th[i].parameters():
                    param += lr_policies_outer[i] * grads[i][k]
                    k += 1
            else:
                th[i] += lr_policies_outer[i] * grads[i]
    return th, losses, G_ts, nl_terms, lola_terms, grad_2_return_1


def dice_update_th(th, vals, n_agents, inner_steps, outer_steps, lr_policies_outer, lr_policies_inner, lr_values, repeat_train_on_same_samples):

    if args.using_nn:
        static_th_copy = th
        static_vals_copy = vals
    else:
        static_th_copy = copy.deepcopy(th)
        static_vals_copy = copy.deepcopy(vals)

    for i in range(n_agents):
        K = inner_steps[i]
        L = outer_steps[i]

        if args.using_nn:
            theta_primes = static_th_copy
            val_primes = static_vals_copy  # should work if it is functional/stateless
        else:
            theta_primes = copy.deepcopy(static_th_copy)
            val_primes = copy.deepcopy(static_vals_copy)

        f_th_primes = []
        if args.using_nn:
            for ii in range(n_agents):
                f_th_primes.append(higher.patch.monkeypatch(theta_primes[ii],
                                                            copy_initial_weights=True,
                                                            track_higher_grads=True))

        mixed_th_lr_policies = copy.deepcopy(lr_policies_inner)
        mixed_th_lr_policies[i] = lr_policies_outer[i]

        optims_th_primes = construct_diff_optims(theta_primes,
                                                 mixed_th_lr_policies,
                                                 f_th_primes)
        f_vals_primes = []
        if args.using_nn:
            for ii in range(n_agents):
                f_vals_primes.append(
                    higher.patch.monkeypatch(
                        val_primes[ii],
                        copy_initial_weights=True,
                        track_higher_grads=True))
        # optims_vals_primes = construct_diff_optims(
        #     val_primes, lr_values * eta,
        #     f_vals_primes)
        optims_vals_primes = construct_diff_optims(
            val_primes, lr_values,
            f_vals_primes)

        if args.using_nn:
            mixed_thetas = f_th_primes
            mixed_vals = f_vals_primes
            # mixed_thetas[i] = th[i]
            # mixed_vals[i] = vals[i]

            # mixed_thetas[i] = f_th[i]
            # mixed_vals[i] = f_vals[i]

        else:
            mixed_thetas = theta_primes
            mixed_vals = val_primes
            mixed_thetas[i] = th[i]
            mixed_vals[i] = vals[i]

        # print(mixed_thetas)
        # print(sum(lr_policies_inner))
        # print(sum(lr_policies_inner)>0)

        if sum(lr_policies_inner) > 0:
            # --- INNER STEPS ---
            if repeat_train_on_same_samples:
                action_trajectory, rewards, policy_history, val_history, next_val_history, obs_history = game.rollout(
                    mixed_thetas, mixed_vals)
                for step in range(K):
                    # print(step)
                    if step == 0:
                        dice_loss, _, values_loss = game.get_dice_loss(
                            action_trajectory, rewards,
                            policy_history, val_history, next_val_history,
                            old_policy_history=policy_history,
                            use_nl_loss=args.inner_nl_loss,
                            use_penalty=args.inner_penalty,
                            use_clipping=args.inner_clip, beta=args.inner_beta)

                    else:
                        new_policies, new_vals, next_new_vals = game.get_policies_vals_for_states(
                            mixed_thetas, mixed_vals, obs_history)
                        # Using the new policies and vals now
                        # Always be careful not to overwrite/reuse names of existing variables
                        dice_loss, _, values_loss = game.get_dice_loss(
                            action_trajectory, rewards, new_policies, new_vals,
                            next_new_vals, old_policy_history=policy_history,
                            use_nl_loss=args.inner_nl_loss,
                            use_penalty=args.inner_penalty,
                            use_clipping=args.inner_clip, beta=args.inner_beta)

                    # grads = [None] * n_agents

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
                                mixed_thetas[j] = mixed_thetas[j] - lr_policies_inner[
                                    j]  * grad  # This step is critical to allow the gradient to flow through
                                # You cannot use torch.no_grad on this step

                                if args.inner_val_updates:
                                    grad_val = get_gradient(
                                        values_loss[j],
                                        mixed_vals[j])
                                    mixed_vals[j] = mixed_vals[j] - lr_values[
                                        j]  * grad_val

                            else:

                                optim_update(optims_th_primes[j],
                                             dice_loss[j],
                                             mixed_thetas[j].parameters())
                                if args.inner_val_updates:
                                    optim_update(optims_vals_primes[j],
                                                 values_loss[j],
                                                 mixed_vals[j].parameters())

                    # Also TODO Aug 23 is do an outer loop with number of steps also

                    if args.print_inner_rollouts:
                        print(
                            "---Agent {} Rollout {}---".format(i + 1, step + 1))
                        game.print_policies_for_all_states(
                            mixed_thetas)
                        game.print_values_for_all_states(mixed_vals)

            # STILL DOING INNER STEPS HERE
            else:  # no repeat train on samples, this is the original DiCE formulation

                for step in range(K):
                    if args.env == 'coin':
                        obs_history, act_history, rewards, policy_history, \
                        val_history, next_val_history, avg_same_colour_coins_picked_total, \
                        avg_diff_colour_coins_picked_total, avg_coins_picked_total = game.rollout(
                            mixed_thetas, mixed_vals)

                        dice_loss, _, values_loss = \
                            game.get_dice_loss(rewards, policy_history,
                                               val_history,
                                               next_val_history,
                                               use_nl_loss=args.inner_nl_loss)
                        # use_penalty=args.inner_penalty,
                        # use_clipping=args.inner_clip,
                        # beta=args.inner_beta)
                    elif args.env == 'imp':
                        obs_history, act_history, rewards, policy_history, \
                        val_history, next_val_history = game.rollout(
                            mixed_thetas, mixed_vals)

                        dice_loss, _, values_loss = game.get_dice_loss(
                            act_history, rewards, policy_history,
                            val_history,
                            next_val_history,
                            use_nl_loss=args.inner_nl_loss)
                        # use_penalty=args.inner_penalty,
                        # use_clipping=args.inner_clip,
                        # beta=args.inner_beta)


                    else:
                        action_trajectory, rewards, policy_history, val_history, next_val_history, obs_history = game.rollout(
                            mixed_thetas, mixed_vals)
                        dice_loss, _, values_loss = game.get_dice_loss(
                            action_trajectory,
                            rewards,
                            policy_history, val_history, next_val_history,
                            use_nl_loss=args.inner_nl_loss)
                        # use_penalty=args.inner_penalty,
                        # use_clipping=args.inner_clip, beta=args.inner_beta)

                    # grads = [None] * n_agents
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
                                mixed_thetas[j] = mixed_thetas[j] - lr_policies_inner[
                                    j] * grad  # This step is critical to allow the gradient to flow through
                                # You cannot use torch.no_grad on this step

                                if args.inner_val_updates:
                                    grad_val = get_gradient(
                                        values_loss[j],
                                        mixed_vals[j])
                                    mixed_vals[j] = mixed_vals[j] - lr_values[
                                        j] * grad_val

                                # TODO The regular DiCE formulation should have value updates on this inner loop too
                                # However either 1) it is hard to choose the right hyperparameter
                                # or 2) you end up using small value updates which don't do much
                                # or 3) you require multiple value update iterations which is going to require more time/make training slower

                            else:

                                optim_update(optims_th_primes[j],
                                             dice_loss[j],
                                             mixed_thetas[j].parameters())

                                if args.inner_val_updates:
                                    optim_update(optims_vals_primes[j],
                                                 values_loss[j],
                                                 mixed_vals[j].parameters())

                    if args.print_inner_rollouts:
                        print("---Agent {} Rollout {}---".format(
                            i + 1, step + 1))
                        game.print_policies_for_all_states(
                            mixed_thetas)
                        game.print_values_for_all_states(
                            mixed_vals)

        # --- OUTER STEPS ---
        # Now calculate outer step using for each player a mix of the theta_primes and old thetas
        # mixed thetas and mixed vals contain the new constructed thetas and vals for all the other players
        # only the agent doing the rollout has the original th/val in the mixed theta/val
        # so then the gradient step afterwards on the original th/val makes sense

        # diff_optims_th = construct_diff_optims(th, lr_policies, f_th)
        # diff_optims_vals = construct_diff_optims(vals, lr_values, f_vals)

        for step in range(L):
            # print(step)
            if repeat_train_on_same_samples:
                if args.env != 'ipd':
                    raise Exception(
                        "Repeat_train not yet supported for other games")
                if step == 0:
                    # This structure is a bit different from the inner loop one... I could make consistent
                    # The key point is you need to roll out once at the beginning and then repeat train afterwards
                    action_trajectory, rewards, policy_history, val_history, next_val_history, obs_history = game.rollout(
                        mixed_thetas, mixed_vals)
                    dice_loss, _, values_loss = game.get_dice_loss(
                        action_trajectory, rewards,
                        policy_history, val_history,
                        next_val_history,
                        old_policy_history=policy_history,
                        use_nl_loss=args.inner_nl_loss,
                        use_penalty=args.outer_penalty,
                        use_clipping=args.outer_clip,
                        beta=args.outer_beta)

                else:
                    new_policies, new_vals, next_new_vals = game.get_policies_vals_for_states(
                        mixed_thetas, mixed_vals,
                        obs_history)
                    # Using the new policies and vals now
                    # Always be careful not to overwrite/reuse names of existing variables
                    dice_loss, _, values_loss = game.get_dice_loss(
                        action_trajectory, rewards,
                        new_policies, new_vals,
                        next_new_vals,
                        old_policy_history=policy_history,
                        use_nl_loss=args.inner_nl_loss,
                        use_penalty=args.outer_penalty,
                        use_clipping=args.outer_clip,
                        beta=args.outer_beta)
            else:
                # No repeat train on same samples
                if args.env == 'coin':
                    obs_history, act_history, rewards, policy_history, val_history, \
                    next_val_history, avg_same_colour_coins_picked_total, avg_diff_colour_coins_picked_total, \
                    avg_coins_picked_total = game.rollout(
                        mixed_thetas, mixed_vals)

                    dice_loss, G_ts, values_loss = game.get_dice_loss(
                        rewards, policy_history, val_history,
                        next_val_history)
                elif args.env == 'imp':
                    if repeat_train_on_same_samples:
                        raise Exception(
                            "Repeat_train not yet supported for coin game")
                    obs_history, act_history, rewards, policy_history, \
                    val_history, next_val_history = game.rollout(
                        mixed_thetas, mixed_vals)

                    dice_loss, _, values_loss = game.get_dice_loss(
                        act_history, rewards, policy_history,
                        val_history,
                        next_val_history)
                    # use_nl_loss=args.inner_nl_loss,
                    # use_penalty=args.outer_penalty,
                    # use_clipping=args.outer_clip,
                    # beta=args.outer_beta)
                else:
                    action_trajectory, rewards, policy_history, val_history, next_val_history, obs_history = game.rollout(
                        mixed_thetas, mixed_vals)

                    # no_penalty_on_outer_step = False
                    # use_pen = args.use_penalty
                    # if no_penalty_on_outer_step:
                    #     use_pen = False

                    # if repeat_train_on_same_samples:
                    #     dice_loss, G_ts, values_loss = game.get_dice_loss(
                    #         action_trajectory, rewards,
                    #         policy_history, val_history, next_val_history,
                    #         old_policy_history=policy_history,
                    #         use_penalty=args.outer_penalty, use_clipping=args.outer_clip)
                    # else:
                    dice_loss, G_ts, values_loss = game.get_dice_loss(
                        action_trajectory, rewards,
                        policy_history, val_history, next_val_history)
                    # use_penalty=args.outer_penalty, use_clipping=args.outer_clip,
                    # beta=args.outer_beta)

            if isinstance(mixed_thetas[i], torch.Tensor):
                # optim_update(optims_th[i], dice_loss[i], [th[i]])
                # optim_update(optims_vals[i], values_loss[i], [vals[i]])

                grad = get_gradient(dice_loss[i], mixed_thetas[i])
                grad_val = get_gradient(values_loss[i], mixed_vals[i])

                with torch.no_grad():
                    mixed_thetas[i] -= lr_policies_outer[i] * grad
                    mixed_vals[i] -= lr_values[i] * grad_val
                    # th[i] -= lr_policies[i] * (b-a)

                    th[i] = mixed_thetas[i]
                    vals[i] = mixed_vals[i]

                    # TODO confirm whether need to copy over to th[i] the new mixed_thetas[i]

                # TODO Be careful with +/- formulation now...
                # TODO rename the non-DiCE terms as rewards
                # And keep the DiCE terms as losses


            else:
                optim_update(optims_th_primes[i], dice_loss[i],
                             mixed_thetas[i].parameters())
                optim_update(optims_vals_primes[i], values_loss[i],
                             mixed_vals[i].parameters())

                copyNN(th[i], mixed_thetas[i])
                copyNN(vals[i], mixed_vals[i])

            for _ in range(args.extra_value_updates):
                _, val_history, next_val_history = game.get_policies_vals_for_states(
                    mixed_thetas, mixed_vals, obs_history)

                if repeat_train_on_same_samples:
                    dice_loss, G_ts, values_loss = game.get_dice_loss(
                        action_trajectory, rewards,
                        policy_history, val_history,
                        next_val_history,
                        old_policy_history=policy_history,
                        use_penalty=args.outer_penalty,
                        use_clipping=args.outer_clip,
                        beta=args.outer_beta)
                else:
                    dice_loss, G_ts, values_loss = game.get_dice_loss(
                        action_trajectory, rewards,
                        policy_history, val_history,
                        next_val_history)
                    # use_penalty=args.outer_penalty, use_clipping=args.outer_clip,
                    # beta=args.outer_beta)

                if isinstance(vals[i], torch.Tensor):
                    grad_val = get_gradient(values_loss[i],
                                            vals[i])
                    with torch.no_grad():
                        vals[i] -= lr_values[i] * grad_val
                else:
                    optim_update(optims_vals[i],
                                 values_loss[i])

            if args.print_outer_rollouts:
                print("---Agent {} Rollout {}---".format(
                    i + 1, step + 1))
                game.print_policies_for_all_states(
                    mixed_thetas)
                game.print_values_for_all_states(
                    mixed_vals)

    return th, vals

def construct_mixed_th_vals_and_diffoptims(n_agents, i, starting_th, starting_vals, lr_policies_outer, lr_policies_inner, lr_values):
    # if args.using_nn:
    #     theta_primes = starting_th
    #     val_primes = starting_vals  # should work if it is functional/stateless
    # else:
    theta_primes = copy.deepcopy(starting_th)
    val_primes = copy.deepcopy(starting_vals)

    # theta_primes, _, val_primes, _, _, _ = init_custom(dims,
    #                                                            args.state_type,
    #                                                            args.using_nn,
    #                                                            args.using_rnn,
    #                                                            args.env,
    #                                                            args.nn_hidden_size,
    #                                                            args.nn_extra_layers)
    # for j in range(n_agents):
    #     copyNN(theta_primes[j], starting_th[j])
    #     copyNN(val_primes[j], starting_vals[j])

    f_th_primes = []
    if args.using_nn:
        for ii in range(n_agents):
            f_th_primes.append(higher.patch.monkeypatch(theta_primes[ii],
                                                        copy_initial_weights=True,
                                                        track_higher_grads=True))

    mixed_th_lr_policies = copy.deepcopy(lr_policies_inner)
    mixed_th_lr_policies[i] = lr_policies_outer[i]

    optims_th_primes = construct_diff_optims(theta_primes,
                                             mixed_th_lr_policies,
                                             f_th_primes)

    f_vals_primes = []
    if args.using_nn:
        for ii in range(n_agents):
            f_vals_primes.append(
                higher.patch.monkeypatch(
                    val_primes[ii],
                    copy_initial_weights=True,
                    track_higher_grads=True))

    optims_vals_primes = construct_diff_optims(
        val_primes, lr_values,
        f_vals_primes)


    if args.using_nn:
        mixed_thetas = f_th_primes
        mixed_vals = f_vals_primes

    else:
        mixed_thetas = theta_primes
        mixed_vals = val_primes

    # optims_th_primes_nodiff = construct_optims(mixed_thetas, mixed_th_lr_policies)
    # optims_vals_primes_nodiff = construct_optims(mixed_vals, lr_values)
    optims_th_primes_nodiff = construct_optims(theta_primes,
                                               mixed_th_lr_policies)
    optims_vals_primes_nodiff = construct_optims(val_primes, lr_values)
    mixed_thetas[i] = theta_primes[i]
    mixed_vals[i] = val_primes[i]

    return mixed_thetas, mixed_vals, optims_th_primes, optims_vals_primes, optims_th_primes_nodiff, optims_vals_primes_nodiff

def dice_inner_step(i, inner_step, action_trajectory, rewards, policy_history,
                    val_history, next_val_history, mixed_thetas, mixed_vals,
                    obs_history, kl_div_target_policy_inner,
                    optims_th_primes, optims_vals_primes):
    if inner_step == 0:
        dice_loss, _, values_loss = game.get_dice_loss(
            action_trajectory, rewards,
            policy_history, val_history, next_val_history,
            old_policy_history=policy_history,
            use_nl_loss=args.inner_nl_loss,
            use_penalty=args.inner_penalty,
            use_clipping=args.inner_clip, beta=args.inner_beta)

    else:
        new_policies, new_vals, next_new_vals = game.get_policies_vals_for_states(
            mixed_thetas, mixed_vals, obs_history)
        # Using the new policies and vals now
        # Always be careful not to overwrite/reuse names of existing variables
        dice_loss, _, values_loss = game.get_dice_loss(
            action_trajectory, rewards, new_policies, new_vals,
            next_new_vals, old_policy_history=policy_history,
            kl_div_target_policy=kl_div_target_policy_inner,
            use_nl_loss=args.inner_nl_loss,
            use_penalty=args.inner_penalty,
            use_clipping=args.inner_clip, beta=args.inner_beta)

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
                mixed_thetas[j] = mixed_thetas[j] - lr_policies_inner[
                    j] * grad  # This step is critical to allow the gradient to flow through
                # You cannot use torch.no_grad on this step

                if args.inner_val_updates:
                    grad_val = get_gradient(
                        values_loss[j],
                        mixed_vals[j])
                    mixed_vals[j] = mixed_vals[j] - lr_policies_inner[
                        j] * grad_val

            else:

                optim_update(optims_th_primes[j],
                             dice_loss[j],
                             mixed_thetas[j].parameters())
                if args.inner_val_updates:
                    optim_update(optims_vals_primes[j],
                                 values_loss[j],
                                 mixed_vals[j].parameters())

    if args.print_inner_rollouts:
        print(
            "---Agent {} Rollout {}---".format(i + 1, inner_step + 1))
        game.print_policies_for_all_states(mixed_thetas)
        game.print_values_for_all_states(mixed_vals)


def dice_update_th_new_repeat_loop(th, vals, n_agents, inner_steps, outer_steps, lr_policies_outer, lr_policies_inner, lr_values):
    # Assumed repeat_train_on_same_samples
    # (and using_nn?)

    # if args.using_nn:
    #     static_th_copy = th
    #     static_vals_copy = vals
    # else:
    static_th_copy = copy.deepcopy(th)
    static_vals_copy = copy.deepcopy(vals)

    # static_th_copy, _, static_vals_copy, _, _, _ = init_custom(dims, args.state_type, args.using_nn, args.using_rnn, args.env, args.nn_hidden_size, args.nn_extra_layers)
    # for i in range(n_agents):
    #     copyNN(static_th_copy[i], th[i])
    #     copyNN(static_vals_copy[i], vals[i])

    for i in range(n_agents):
        K = inner_steps[i]
        L = outer_steps[i]

        # print(mixed_thetas)
        if sum(lr_policies_inner) > 0:
            assert repeat_train_on_same_samples

            for outer_step in range(L):

                th_with_only_agent_i_updated = copy.deepcopy(static_th_copy)
                th_with_only_agent_i_updated[i] = th[i]
                vals_with_only_agent_i_updated = copy.deepcopy(static_vals_copy)
                vals_with_only_agent_i_updated[i] = vals[i]

                # mixed_thetas, mixed_vals, optims_th_primes, optims_vals_primes, \
                # optims_th_primes_nodiff, optims_vals_primes_nodiff = \
                #     construct_mixed_th_vals_and_diffoptims(n_agents, i,
                #                                            static_th_copy,
                #                                            static_vals_copy,
                #                                            lr_policies,
                #                                            lr_values, eta)

                # Reconstruct on ever outer loop iter. The idea is this:
                # Starting from the static th, take steps updating all the other player policies for x inner steps (roughly until convegence or doesn't have to be)
                # Then update own policy, ONCE
                # Then we repeat, starting from the static th for all other players' policies
                # BUT now we have the old policies of all other players but our own updated policy
                # Then the other players again solve the prox objective, but with our new policy
                # And then we take another step
                # And repeat
                # Starting from the old th for all other players again and diff through those steps
                # Is analogous to what the IFT does after we have found a fixed point
                # We can't warm start here though if I am resolving/recreating the optims/gradient tape at each outer step
                # And the reason I do this resolve instead of just one long gradient tape
                # Is because otherwise the memory blows up way too quickly
                mixed_thetas, mixed_vals, optims_th_primes, optims_vals_primes, \
                optims_th_primes_nodiff, optims_vals_primes_nodiff = \
                    construct_mixed_th_vals_and_diffoptims(n_agents, i,
                                                           th_with_only_agent_i_updated,
                                                           vals_with_only_agent_i_updated,
                                                           lr_policies_outer, lr_policies_inner,
                                                           lr_values)

                # print("HEY")
                # game.print_policies_for_all_states(static_th_copy)
                # game.print_policies_for_all_states(th_with_only_agent_i_updated)
                # game.print_policies_for_all_states(mixed_thetas)



                # INNER LOOP

                # if outer_step == 0:
                #     mixed_thetas, mixed_vals, optims_th_primes, optims_vals_primes = \
                #         construct_mixed_th_vals_and_diffoptims(n_agents, i,
                #                                                static_th_copy,
                #                                                static_vals_copy,
                #                                                lr_policies,
                #                                                lr_values, eta)
                # else:
                #     # No I don't think you can do this. Well you can but then by resetting gradient tape
                #     # The later inner steps won't really have much gradient signal for the higher order gradient
                #     mixed_thetas, mixed_vals, optims_th_primes, optims_vals_primes = \
                #         construct_mixed_th_vals_and_diffoptims(n_agents, i,
                #                                                mixed_thetas,
                #                                                mixed_vals,
                #                                                lr_policies,
                #                                                lr_values, eta)




                action_trajectory, rewards, policy_history, val_history, next_val_history, obs_history = game.rollout(
                    mixed_thetas, mixed_vals)
                if outer_step == 0:
                    kl_div_target_policy_inner = policy_history.detach().clone()

                for inner_step in range(K):

                    # print(inner_step)
                    dice_inner_step(i, inner_step, action_trajectory, rewards,
                                        policy_history,
                                        val_history, next_val_history,
                                        mixed_thetas, mixed_vals,
                                        obs_history, kl_div_target_policy_inner,
                                        optims_th_primes, optims_vals_primes)


                # --- OUTER STEP ---

                # print(step)
                if args.env != 'ipd':
                    raise Exception(
                        "Repeat_train not yet supported for other games")

                action_trajectory, rewards, policy_history, val_history, next_val_history, obs_history = game.rollout(
                    mixed_thetas, mixed_vals)
                if outer_step == 0:
                    kl_div_target_policy_outer = policy_history.detach().clone()

                dice_loss, _, values_loss = game.get_dice_loss(
                    action_trajectory, rewards,
                    policy_history, val_history,
                    next_val_history,
                    old_policy_history=policy_history,
                    kl_div_target_policy=kl_div_target_policy_outer,
                    use_nl_loss=args.inner_nl_loss,
                    use_penalty=args.outer_penalty,
                    use_clipping=args.outer_clip,
                    beta=args.outer_beta)

                # If no new rollout
                # new_policies, new_vals, next_new_vals = game.get_policies_vals_for_states(
                #     mixed_thetas, mixed_vals,
                #     obs_history)
                # # Using the new policies and vals now
                # # Always be careful not to overwrite/reuse names of existing variables
                # dice_loss, _, values_loss = game.get_dice_loss(
                #     action_trajectory, rewards,
                #     new_policies, new_vals,
                #     next_new_vals,
                #     old_policy_history=policy_history,
                #     use_nl_loss=args.inner_nl_loss,
                #     use_penalty=args.outer_penalty,
                #     use_clipping=args.outer_clip,
                #     beta=args.outer_beta)


                if isinstance(mixed_thetas[i], torch.Tensor):
                    # optim_update(optims_th[i], dice_loss[i], [th[i]])
                    # optim_update(optims_vals[i], values_loss[i], [vals[i]])

                    grad = get_gradient(dice_loss[i], mixed_thetas[i])
                    grad_val = get_gradient(values_loss[i], mixed_vals[i])

                    with torch.no_grad():
                        mixed_thetas[i] -= lr_policies_outer[i] * grad
                        mixed_vals[i] -= lr_values[i] * grad_val
                        # th[i] -= lr_policies_outer[i] * (b-a)
                        th[i] = mixed_thetas[i]
                        vals[i] = mixed_vals[i]

                        1 / 0
                        # TODO confirm whether need to copy over to th[i] the new mixed_thetas[i]

                    # TODO Be careful with +/- formulation now...
                    # TODO rename the non-DiCE terms as rewards
                    # And keep the DiCE terms as losses


                else:
                    # game.print_policies_for_all_states(
                    #     mixed_thetas)
                    optim_update(optims_th_primes_nodiff[i], dice_loss[i],)
                    optim_update(optims_vals_primes_nodiff[i], values_loss[i],)
                    # optim_update(optims_th_primes[i], dice_loss[i],
                    #              mixed_thetas[i].parameters())
                    # optim_update(optims_vals_primes[i], values_loss[i],
                    #              mixed_vals[i].parameters())

                    # game.print_policies_for_all_states(
                    #     mixed_thetas)
                    # 1/0

                    copyNN(th[i], mixed_thetas[i])
                    copyNN(vals[i], mixed_vals[i])

                for _ in range(args.extra_value_updates):
                    _, val_history, next_val_history = game.get_policies_vals_for_states(
                        mixed_thetas, mixed_vals, obs_history)

                    dice_loss, G_ts, values_loss = game.get_dice_loss(
                        action_trajectory, rewards,
                        policy_history, val_history,
                        next_val_history,
                        old_policy_history=policy_history,
                        use_penalty=args.outer_penalty,
                        use_clipping=args.outer_clip,
                        beta=args.outer_beta)

                    if isinstance(vals[i], torch.Tensor):
                        grad_val = get_gradient(values_loss[i],
                                                vals[i])
                        with torch.no_grad():
                            vals[i] -= lr_values[i] * grad_val
                    else:
                        optim_update(optims_vals[i],
                                     values_loss[i])

                if args.print_outer_rollouts:
                    print("---Agent {} Rollout {}---".format(
                        i + 1, outer_step + 1))
                    game.print_policies_for_all_states(
                        mixed_thetas)
                    game.print_values_for_all_states(
                        mixed_vals)

    return th, vals




if __name__ == "__main__":
    parser = argparse.ArgumentParser("NPLOLA")
    parser.add_argument("--env", type=str, default="ipd",
                        # choices=["ipd", "coin", "hawkdove"])
                        choices=["ipd", "coin", "imp"])
    parser.add_argument("--state_type", type=str, default="one_hot",
                        choices=['mnist', 'one_hot', 'majorTD4', 'old'],
                        help="For IPD/social dilemma, choose the state/obs representation type. One hot is the default. MNIST feeds in MNIST digits (0 or 1) instead of one hot class 0, class 1, etc. Old is there to support original/old formulation where we just had state representation 0,1,2. This is fine with tabular but causes issues with function approximation (where since 1 is coop, 2 is essentially 'super coop')")
    parser.add_argument("--using_samples", action="store_true",
                        help="True for samples (with rollout_len), false for exact gradient (using matrix inverse for infinite length rollout)")
    parser.add_argument("--using_DiCE", action="store_true",
                        help="True for LOLA-DiCE, false for LOLA-PG. Must have using_samples = True.")
    parser.add_argument("--repeat_train_on_same_samples", action="store_true",
                        help="True for PPO style formulation where we repeat train on the same samples (only one inner step rollout, multiple inner step updates with importance weighting)")
    # parser.add_argument("--use_clipping", action="store_true",
    #                     help="Do the PPO style clipping")
    parser.add_argument("--inner_penalty", action="store_true",
                        help="Apply PPO style adaptive KL penalty on inner loop steps")
    parser.add_argument("--outer_penalty", action="store_true",
                        help="Apply PPO style adaptive KL penalty on outer loop steps")
    parser.add_argument("--inner_clip", action="store_true",
                        help="Apply PPO style clipping on inner loop steps")
    parser.add_argument("--outer_clip", action="store_true",
                        help="Apply PPO style clipping on outer loop steps")
    parser.add_argument("--clip_epsilon", type=float, default=0.2, help="PPO style clip hyperparameter")
    parser.add_argument("--gamma", type=float, default=0.96, help="discount rate")
    parser.add_argument("--print_every", type=int, default=200, help="Print every x number of epochs")
    parser.add_argument("--num_epochs", type=int, default=50001, help="number of epochs to run")
    parser.add_argument("--repeats", type=int, default=1, help="repeats per setting configuration")
    parser.add_argument("--n_agents_list", nargs="+", type=int, default=[5],
                        help="list of number of agents to try")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr_policies_outer", type=float, default=0.05,
                        help="outer loop learning rate: same learning rate across all policies for now")
    parser.add_argument("--lr_policies_inner", type=float, default=0.05,
                        help="inner loop learning rate: this has no use in the naive learning case. Used for the gradient step done for the lookahead for other agents during LOLA (therefore, often scaled to be higher than the outer learning rate in non-proximal LOLA). This is the eta step size in the original LOLA paper. For prox, it is the learning rate on the inner prox loop.")
    parser.add_argument("--lr_values", type=float, default=0.025,
                        help="same learning rate across all policies for now. Should be around maybe 0.001 or less for neural nets to avoid instability")
    parser.add_argument("--inner_steps", type=int, default=1, help="inner loop steps for DiCE")
    parser.add_argument("--outer_steps", type=int, default=1)
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
    # parser.add_argument("--mnist_states", action="store_true",
    #                     help="use MNIST digits as state representation") # Deprecated, see state_type
    parser.add_argument("--init_state_representation", type=int, default=2)
    parser.add_argument("--rollout_len", type=int, default=50, help="How long we want the time horizon of the game to be (number of steps before termination/number of iterations of the IPD)")
    parser.add_argument("--using_rnn", action="store_true",
                        help="use RNN (for coin game)") # TODO only supported for coin right now
    parser.add_argument("--inner_nl_loss", action="store_true",
                        help="use naive learning (no shaping) loss on inner dice loop")
    parser.add_argument("--inner_val_updates", action="store_true",
                        help="value updates on the inner dice loop")
    parser.add_argument("--two_way_clip", action="store_true",
                        help="use 2 way clipping instead of PPO clip")
    parser.add_argument("--base_cf_no_scale", type=float, default=1.6,
                        help="base contribution factor for no scaling (right now for 2 agents)")
    parser.add_argument("--base_cf_scale", type=float, default=0.6,
                        help="base contribution factor with scaling (right now for >2 agents)")
    parser.add_argument("--std", type=float, default=0.1, help="standard deviation for initialization of policy/value parameters")
    parser.add_argument("--inner_beta", type=float, default=1, help="beta determines how strong we want the KL penalty to be. Use either with --inner_exact_prox (exact gradients) or --inner_penalty (func approx)")
    parser.add_argument("--outer_beta", type=float, default=1, help="beta determines how strong we want the KL penalty to be. Use either with --outer_exact_prox (exact gradients) or --outer_penalty (func approx)")
    parser.add_argument("--print_inner_rollouts", action="store_true")
    parser.add_argument("--print_outer_rollouts", action="store_true")
    parser.add_argument("--inner_exact_prox", action="store_true",
                        help="find exact prox solution in inner loop instead of x # of inner steps")
    parser.add_argument("--outer_exact_prox", action="store_true",
                        help="find exact prox solution in outer loop instead of x # of outer steps")
    parser.add_argument("--no_taylor_approx", action="store_true",
                        help="experimental: try DiCE style, direct update of policy and diff through it")
    parser.add_argument("--ill_condition", action="store_true",
                        help="in exact case, add preconditioning to make the problem ill-conditioned. Used to test if prox-lola helps")
    # parser.add_argument("--ill_cond_diag_matrix", nargs="+", type=float, default=[3., 0.1, 0.1, 0.1, 1.],
    #                     help="The ill conditioning matrix (diagonal entries) to use for the ill_condition. Dim should match dims[0] (i.e. 2^n + 1)")
    parser.add_argument("--print_prox_loops_info", action="store_true",
                        help="print some additional info for the prox loop iters")
    parser.add_argument("--prox_threshold", type=float, default=1e-8, help="Threshold for KL divergence below which we consider a fixed point to have been reached for the proximal LOLA. Recommended not to go to higher than 1e-8 if you want something resembling an actual fixed point")
    parser.add_argument("--prox_max_iters", type=int, default=5000, help="Maximum proximal steps to take before timeout")
    parser.add_argument("--dd_stretch_factor", type=float, default=2., help="for ill conditioning in the func approx case, stretch logit of policy in DD state by this amount")
    parser.add_argument("--all_state_stretch_factor", type=float, default=0.33, help="for ill conditioning in the func approx case, stretch logit of policy in all states by this amount")
    parser.add_argument("--theta_init_mode", type=str, default="standard",
                        choices=['standard', 'tft', 'adv', 'adv2'],
                        help="For IPD/social dilemma in the exact gradient/tabular setting, choose the policy initialization mode.")
    parser.add_argument("--exact_grad_calc", action="store_true",
                        help="Only calc exact gradients, don't run the algo")
    parser.add_argument("--dice_grad_calc", action="store_true",
                        help="Only calc DiCE gradients, don't run the algo")
    parser.add_argument("--exact_finite_horizon", action="store_true",
                        help="Use limited horizon (rollout_len) for the exact gradient case")
    parser.add_argument("--mnist_coop_class", type=int, default=1, help="Digit class to use in place of the observation when an agent cooperates, when using MNIST state representation")
    parser.add_argument("--mnist_defect_class", type=int, default=0, help="Digit class to use in place of the observation when an agent defects, when using MNIST state representation")


    args = parser.parse_args()

    torch.autograd.set_detect_anomaly(True)

    init_state_representation = args.init_state_representation

    rollout_len = args.rollout_len

    if args.set_seed:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    std = args.std

    # Repeats for each hyperparam setting
    # repeats = 10
    repeats = args.repeats

    if args.outer_exact_prox:
        assert args.inner_exact_prox or args.no_taylor_approx

    if args.inner_penalty or args.outer_penalty or args.inner_clip or args.outer_clip:
        assert args.repeat_train_on_same_samples # Otherwise these don't do anything if you just take 1 step on samples

    if args.repeat_train_on_same_samples:
        assert args.using_DiCE


    if args.dice_grad_calc:
        assert args.using_DiCE and args.using_samples # and args.using_nn


    # tanh instead of relu or lrelu activation seems to help. Perhaps the gradient flow is a bit nicer that way

    if args.ill_condition:
        # ill_cond_matrix = torch.diag(torch.tensor(args.ill_cond_diag_matrix))

        # ill_cond_matrix = torch.tensor([[-2., -1., 0., 0., 0.],
        #                                 [0., 0.1, -0.1, 0., 0.],
        #                                 [0., 0., -0.1, -0.5, 0.],
        #                                 [0., 0., 0., -0.3, -1.],
        #                                 [0., 0., 0., 0., -1.]])
        # ill_cond_matrix = torch.tensor([[1., 1./2., 1./3., 1./4., 1./5.],
        #                                 [1./2., 1./3., 1./4., 1./5., 1./6.],
        #                                 [1./3., 1./4., 1./5., 1./6., 1./7.],
        #                                 [1./4., 1./5., 1./6., 1./7., 1./8.],
        #                                 [1./5., 1./6., 1./7., 1./8., 1./9.]])
        # ill_cond_matrix = torch.tensor([[2., 1.8, 1.6, 1.4, 1.2],
        #                                 [1.8, 1.6, 1.4, 1.2, 1.],
        #                                 [1.6, 1.4, 1.2, 1., 0.8],
        #                                 [1.4, 1.2, 1., 0.8, 0.6],
        #                                 [1.2, 1., 0.8, 0.6, 0.4]]) # this is not full rank...
        # ill_cond_matrix = torch.tensor([[2., 1.6, 1.2, 0.8, 0.4],
        #                                 [1.6, 1.3, 1., 0.7, 0.4],
        #                                 [1.2, 1., 0.8, 0.6, 0.4],
        #                                 [0.8, 0.7, 0.6, 0.5, 0.4],
        #                                 [0.4, 0.4, 0.4, 0.4, 0.4]]) / 2. # also singular...
        # ill_cond_matrix = torch.tensor([[2., 1.8, 1., 0.6, 0.2],
        #                                 [1.6, 1.6, 1., 1., 0.4],
        #                                 [1.2, 1.4, 1., 1.2, 0.8],
        #                                 [0.8, 1.2, 1., 1.6, 1.6],
        #                                 [0.4, 1., 1., 1.8, 2.0]])
        # print("IF USING FUNC APPROX DO NOT EDIT THIS MATRIX, EDIT THE SECTION IN THE CONTRIB GAME")
        # ill_cond_matrix = torch.tensor([[3., 0., 0., 0., 0.],
        #                                 [0., 0.1, 0, 0., 0.],
        #                                 [0., 0., 0.1, 0, 0.],
        #                                 [0., 0., 0., 0.1, 0.],
        #                                 [0., 0., 0., 0., 1.]])
        # ill_cond_matrix = torch.tensor([[1., 0., 0., 0.9, 0.],
        #                                 [0., 1, 0.9, 0., 0.],
        #                                 [0., 0.9, 1., 0, 0.],
        #                                 [0.9, 0., 0., 1., 0.],
        #                                 [0., 0., 0., 0., 1.]]) # this isn't bad condition, just confuses states a lot
        # ill_cond_matrix = torch.tensor([[.5, 0., 0., 0, 0.],
        #                                 [.5, .1, 0, 0., 0.],
        #                                 [.5, 0, .1, 0, 0.],
        #                                 [.5, 0., 0., .1, 0.],
        #                                 [.5, 0., 0., 0., 1.]])

        # ill_cond_matrix = torch.tensor([[.2, 1., 0., 0, 0.],
        #                                 [0., 1, 0, 0., 0.],
        #                                 [0, 1., .2, 0, 0.],
        #                                 [0, 1., 0., .2, 0.],
        #                                 [0, 1., 0., 0., 1.]])
        # ill_cond_matrices = torch.stack((ill_cond_matrix, ill_cond_matrix)) # hardcoded 2 agents for now

        # ill_cond_matrix1 = torch.tensor([[.2, 0, 1., 0, 0.],
        #                                  [0., .2, 1, 0., 0.],
        #                                  [0, 0., 1, 0, 0.],
        #                                  [0, 0., 1., .2, 0.],
        #                                  [0, 0., 1., 0., 1.]])
        # ill_cond_matrix2 = torch.tensor([[.2, 1., 0., 0, 0.],
        #                                  [0., 1, 0, 0., 0.],
        #                                  [0, 1., .2, 0, 0.],
        #                                  [0, 1., 0., .2, 0.],
        #                                  [0, 1., 0., 0., 1.]])
        # ill_cond_matrix1 = torch.tensor([[.1, 0, 3., 0, 0.],
        #                                  [0., .1, 3, 0., 0.],
        #                                  [0, 0., 3, 0, 0.],
        #                                  [0, 0., 3., .1, 0.],
        #                                  [0, 0., 3., 0., 1.]])
        # ill_cond_matrix2 = torch.tensor([[.1, 3., 0., 0, 0.],
        #                                  [0., 3, 0, 0., 0.],
        #                                  [0, 3., .1, 0, 0.],
        #                                  [0, 3., 0., .1, 0.],
        #                                  [0, 3., 0., 0., 1.]])
        ill_cond_matrix1 = torch.tensor([[.14, 0, 1., 0, 0.],
                                         [0., .14, 1, 0., 0.],
                                         [0, 0., 1, 0, 0.],
                                         [0, 0., 1., .14, 0.],
                                         [0, 0., 1., 0., 1.]])
        ill_cond_matrix2 = torch.tensor([[.14, 1., 0., 0, 0.],
                                         [0., 1, 0, 0., 0.],
                                         [0, 1., .14, 0, 0.],
                                         [0, 1., 0., .14, 0.],
                                         [0, 1., 0., 0., 1.]])
        # ill_cond_matrix1 = torch.tensor([[1, 2, 0., 0, 0.],
        #                                  [2., 1, 0, 0., 0.],
        #                                  [0, 0., 1, 2, 0.],
        #                                  [0, 0., 2, 1, 0.],
        #                                  [0, 0., 0., 0., 1.]])
        # ill_cond_matrix2 = torch.tensor([[1, 0, 2, 0, 0.],
        #                                  [0., 1, 0, 2., 0.],
        #                                  [2, 0., 1, 0, 0.],
        #                                  [0, 2, 0., 1, 0.],
        #                                  [0, 0., 0., 0., 1.]])
        # ill_cond_matrix1 = torch.tensor([[1, 1, 1, 1, 1.],
        #                                  [0., 1, 1, 1., 1.],
        #                                  [0, 0., 1, 1, 1.],
        #                                  [0, 0, 0., 1, 1.],
        #                                  [0, 0., 0., 0., 1.]])
        # ill_cond_matrix2 = ill_cond_matrix1
        # ill_cond_matrix1 = torch.tensor([[1, 0, 0, 0, 0.],
        #                                  [1., 1, 0, 0., 0.],
        #                                  [1, 1., 1, 0, 0.],
        #                                  [1, 1, 1., 1, 0.],
        #                                  [1, 1., 1., 1., 1.]])
        # ill_cond_matrix2 = ill_cond_matrix1
        # ill_cond_matrix1 = torch.tensor([[1, 0., 0., 0, 0.],
        #                                 [1, .1, 0, 0., 0.],
        #                                 [1, 0, .1, 0, 0.],
        #                                 [1, 0., 0., .1, 0.],
        #                                 [1, 0., 0., 0., 1.]])
        # ill_cond_matrix2 = ill_cond_matrix1
        ill_cond_matrices = torch.stack((ill_cond_matrix1, ill_cond_matrix2)) # hardcoded 2 agents for now

        # torch.eye(dims[0]) @
        print(ill_cond_matrices[0])
        print(ill_cond_matrices[1])
        # print(torch.inverse(ill_cond_matrix))

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
        clip_epsilon = args.clip_epsilon
        two_way_clip = args.two_way_clip
    # TODO it seems the non-DiCE version with batches isn't really working.

    if args.history_len > 1:
        assert args.using_nn # Right now only supported for func approx.

    # mnist_states = args.mnist_states


    n_agents_list = args.n_agents_list
    # n_agents_list = [5, 8]

    if args.env != "ipd":
        if not using_samples:
            raise NotImplementedError("No exact gradient calcs done for this env yet")
        if not args.using_nn:
            raise NotImplementedError("No tabular built for this env yet")


    for n_agents in n_agents_list:

        if args.ill_condition:
            assert n_agents == 2 # Other agents not yet supported for this
            assert args.history_len == 1 # longer hist len not yet supported

        start = timer()

        assert n_agents >= 2
        if n_agents == 2:
            contribution_factor = args.base_cf_no_scale #1.6
            contribution_scale = False
        else:
            contribution_factor = args.base_cf_scale #0.6
            contribution_scale = True

        if batch_size == n_agents or batch_size == rollout_len or rollout_len == n_agents:
            raise Exception("Having two of batch size, rollout len, or n_agents equal will cause insidious bugs when reshaping dimensions")
            # TODO refactor/think about a way to avoid these bugs

        lr_policies_outer = torch.tensor([args.lr_policies_outer] * n_agents)
        lr_policies_inner = torch.tensor([args.lr_policies_inner] * n_agents)

        lr_values = torch.tensor([args.lr_values] * n_agents)

        # if using_DiCE:
        #     etas = args.etas  # [8] # [20] # this is a factor by which we increase the lr on the inner loop vs outer loop

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


        print("Number of agents: {}".format(n_agents))
        print("Contribution factor: {}".format(contribution_factor))
        print("Scaled contribution factor? {}".format(contribution_scale))
        # print("Eta: {}".format(eta))
        # print(reward_percent_of_max)
        # Average over all runs

        if args.exact_grad_calc:
            total_is_in_tft_direction_p1 = 0
            total_is_in_tft_direction_p2 = 0

            for iter in range(num_epochs):

                dims, Ls = ipdn(n=n_agents, gamma=gamma,
                                contribution_factor=contribution_factor,
                                contribution_scale=contribution_scale)
                th = init_th_uniform(dims)  # TODO init uniform
                # th = init_th_adversarial3(dims)
                is_in_tft_direction_p1, is_in_tft_direction_p2 = exact_grad_calc(th, Ls)
                total_is_in_tft_direction_p1 += is_in_tft_direction_p1
                total_is_in_tft_direction_p2 += is_in_tft_direction_p2
                print("% TFT direction for LOLA terms")
                print(total_is_in_tft_direction_p1 / (iter + 1))
                print(total_is_in_tft_direction_p2 / (iter + 1))
                print("P1: {}".format(total_is_in_tft_direction_p1))
                print("P2: {}".format(total_is_in_tft_direction_p2))
                print("Iters: {}".format(iter+1))

            1/0



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
                # Exact gradient case

                dims, Ls = ipdn(n=n_agents, gamma=gamma,
                                contribution_factor=contribution_factor,
                                contribution_scale=contribution_scale)


                # std = 0.1
                if args.theta_init_mode == 'tft':
                    # std = 0.1
                    # Basically with std higher, you're going to need higher logit shift (but only slightly, really), in order to reduce the variance
                    # and avoid random-like behaviour which could undermine the closeness/pull into the TFT basin of solutions
                    th = init_th_tft(dims, std, logit_shift=1.7)
                    # Need around 1.85 for NL and 1.7 for LOLA
                elif args.theta_init_mode == 'adv':
                    th = init_th_adversarial(dims)
                elif args.theta_init_mode == 'adv2':
                    th = init_th_adversarial2(dims)
                else:
                    th = init_th(dims, std)


            else:
                # Using samples instead of exact here



                if args.env == "coin":
                    from coin_game import CoinGameVec
                    # 150 was their default in the alshedivat repo. But they did that for IPD too, which is not really necessary given the high-ish discount rate
                    game = CoinGameVec(max_steps=rollout_len, batch_size=batch_size,
                                       history_len=args.history_len, full_seq_obs=args.using_rnn)
                    dims = game.dims_with_history

                elif args.env == "hawkdove":
                    game = HawkDoveGame(n=n_agents, gamma=gamma,
                                            batch_size=batch_size,
                                            num_iters=rollout_len,
                                            contribution_factor=contribution_factor,
                                            contribution_scale=contribution_scale,
                                            history_len=args.history_len,
                                            using_mnist_states=args.mnist_states)
                    dims = game.dims
                elif args.env == "imp":
                    from matching_pennies import IteratedMatchingPennies
                    game = IteratedMatchingPennies(n=n_agents, batch_size=batch_size,
                                                num_iters=rollout_len, history_len=args.history_len)
                    dims = game.dims

                else:
                    game = ContributionGame(n=n_agents, gamma=gamma,
                                            batch_size=batch_size,
                                            num_iters=rollout_len,
                                            contribution_factor=contribution_factor,
                                            contribution_scale=contribution_scale,
                                            history_len=args.history_len,
                                            state_type=args.state_type)
                    dims = game.dims

                th, vals, optims_vals = init_custom(dims, args.state_type, args.using_nn, args.using_rnn, args.env)

                # I think part of the issue is if policy saturates at cooperation it never explores and never tries defect
                # How does standard reinforce/policy gradient get around this? Entropy regularization
                # Baseline might help too. As might negative rewards everywhere.


            if using_DiCE:
                inner_steps = [args.inner_steps] * n_agents
                outer_steps = [args.outer_steps] * n_agents

            else:
                # algos = ['nl', 'lola']
                # algos = ['lola', 'nl']
                # NOTE: I had previously tested mixed algorithms, e.g. nl with lola
                # However I have not tested this in recent iterations, so be careful if changing this to not be uniform among agents
                # My current general strategy is NL = LOLA with 0 eta (0 inner learning rate on policy, or 0 inner steps)
                algos = ['lola'] * n_agents


            # Run
            if using_samples:
                G_ts_record = torch.zeros((num_epochs, n_agents, batch_size, 1))
            else:
                G_ts_record = torch.zeros(
                    (num_epochs, n_agents))
            # lola_terms_running_total = []
            # nl_terms_running_total = []

            if args.dice_grad_calc:
                total_is_in_tft_direction_p1 = 0
                total_is_in_tft_direction_p2 = 0
                for epoch in range(num_epochs):
                    # Re-init each time, can use higher std for more variance here
                    th, vals, _ = init_custom(dims, args.state_type, args.using_nn,
                        args.using_rnn, args.env)

                    static_th_copy = copy.deepcopy(th)
                    static_vals_copy = copy.deepcopy(vals)

                    nl_updates = []
                    lola_updates = []

                    for i in range(n_agents):
                        nl_th_copy = copy.deepcopy(static_th_copy)
                        nl_vals_copy = copy.deepcopy(static_vals_copy)

                        optims_nl_th = construct_optims(nl_th_copy, lr_policies_outer)

                        action_trajectory, rewards, policy_history, val_history, next_val_history, obs_history = game.rollout(
                            nl_th_copy, nl_vals_copy)

                        nl_loss, _, values_loss = game.get_dice_loss(
                            action_trajectory, rewards,
                            policy_history, val_history,
                            next_val_history,
                            old_policy_history=policy_history,
                            use_nl_loss=True,
                            use_penalty=args.outer_penalty,
                            use_clipping=args.outer_clip,
                            beta=args.outer_beta)

                        # Actually you can't multiple back prop in any case because
                        # The th/policy of each agent
                        # influences the loss of every other agent
                        # so after backprop any th value
                        # you can no longer backprop again (unless you differentiate through the other update, which isn't what we want either)
                        # Will JAX let you separate the gradient calc from the update step?
                        optim_update(optims_nl_th[i], nl_loss[i])

                        nl_updated_policy_i = game.get_policy_for_all_states(nl_th_copy, i)

                        nl_update = nl_updated_policy_i - game.get_policy_for_all_states(static_th_copy, i)
                        nl_updates.append(nl_update.detach())


                    for i in range(n_agents):
                        K = inner_steps[i]
                        L = outer_steps[i]

                        # print(mixed_thetas)
                        if sum(lr_policies_inner) > 0:
                            for outer_step in range(L):
                                th_with_only_agent_i_updated = copy.deepcopy(
                                    static_th_copy)
                                th_with_only_agent_i_updated[i] = th[i]
                                vals_with_only_agent_i_updated = copy.deepcopy(
                                    static_vals_copy)
                                vals_with_only_agent_i_updated[i] = vals[i]

                                mixed_thetas, mixed_vals, optims_th_primes, optims_vals_primes, \
                                optims_th_primes_nodiff, optims_vals_primes_nodiff = \
                                    construct_mixed_th_vals_and_diffoptims(
                                        n_agents, i,
                                        th_with_only_agent_i_updated,
                                        vals_with_only_agent_i_updated,
                                        lr_policies_outer, lr_policies_inner,
                                        lr_values)

                                action_trajectory, rewards, policy_history, val_history, next_val_history, obs_history = game.rollout(
                                    mixed_thetas, mixed_vals)
                                if outer_step == 0:
                                    kl_div_target_policy_inner = policy_history.detach().clone()

                                for inner_step in range(K):

                                    # print(inner_step)
                                    dice_inner_step(i, inner_step,
                                                    action_trajectory,
                                                    rewards,
                                                    policy_history,
                                                    val_history,
                                                    next_val_history,
                                                    mixed_thetas,
                                                    mixed_vals,
                                                    obs_history,
                                                    kl_div_target_policy_inner,
                                                    optims_th_primes,
                                                    optims_vals_primes)

                                # --- OUTER STEP ---

                                # print(step)
                                if args.env != 'ipd':
                                    raise Exception(
                                        "Repeat_train not yet supported for other games")

                                action_trajectory, rewards, policy_history, val_history, next_val_history, obs_history = game.rollout(
                                    mixed_thetas, mixed_vals)
                                if outer_step == 0:
                                    kl_div_target_policy_outer = policy_history.detach().clone()

                                dice_loss, _, values_loss = game.get_dice_loss(
                                    action_trajectory, rewards,
                                    policy_history, val_history,
                                    next_val_history,
                                    old_policy_history=policy_history,
                                    kl_div_target_policy=kl_div_target_policy_outer,
                                    use_nl_loss=args.inner_nl_loss,
                                    use_penalty=args.outer_penalty,
                                    use_clipping=args.outer_clip,
                                    beta=args.outer_beta)

                                if isinstance(mixed_thetas[i],
                                              torch.Tensor):
                                    grad = get_gradient(dice_loss[i],
                                                        mixed_thetas[i])

                                    with torch.no_grad():
                                        mixed_thetas[i] -= lr_policies_outer[
                                                               i] * grad

                                        th[i] = mixed_thetas[i]

                                else:

                                    optim_update(optims_th_primes_nodiff[i],
                                                 dice_loss[i], )
                                    # optim_update(
                                    #     optims_vals_primes_nodiff[i],
                                    #     values_loss[i], )

                                    copyNN(th[i], mixed_thetas[i])
                                    # copyNN(vals[i], mixed_vals[i])


                                # Naive gradient comparison
                                th_with_only_agent_i_updated = copy.deepcopy(
                                    static_th_copy)
                                th_with_only_agent_i_updated[i] = th[i]
                                vals_with_only_agent_i_updated = copy.deepcopy(
                                    static_vals_copy)
                                vals_with_only_agent_i_updated[i] = vals[i]

                                mixed_thetas, mixed_vals, optims_th_primes, optims_vals_primes, \
                                optims_th_primes_nodiff, optims_vals_primes_nodiff = \
                                    construct_mixed_th_vals_and_diffoptims(
                                        n_agents, i,
                                        th_with_only_agent_i_updated,
                                        vals_with_only_agent_i_updated,
                                        lr_policies_outer, lr_policies_inner,
                                        lr_values)

                                action_trajectory, rewards, policy_history, val_history, next_val_history, obs_history = game.rollout(
                                    mixed_thetas, mixed_vals)

                                lola_updated_policy_i = game.get_policy_for_all_states(mixed_thetas, i)
                                lola_update = lola_updated_policy_i - game.get_policy_for_all_states(
                                    static_th_copy, i)
                                lola_updates.append(lola_update.detach())

                    # print(nl_updated_policies)
                    # print(lola_updated_policies)
                    for i in range(n_agents):
                        print("Agent {}".format(i+1))
                        print(lola_updates[i])
                        print(nl_updates[i])
                        print(lola_updates[i] - nl_updates[i])

                    lola_terms_p1 = lola_updates[0] - nl_updates[0]
                    lola_terms_p2 = lola_updates[1] - nl_updates[1]
                    is_in_tft_direction_p1, is_in_tft_direction_p2 = check_is_in_tft_direction(lola_terms_p1, lola_terms_p2)
                    total_is_in_tft_direction_p1 += is_in_tft_direction_p1
                    total_is_in_tft_direction_p2 += is_in_tft_direction_p2
                    print(total_is_in_tft_direction_p1)
                    print(total_is_in_tft_direction_p2)
                    print(epoch+1)

                exit()
                1/0



            for epoch in range(num_epochs):
                if epoch == 0:
                    print("Batch size: " + str(batch_size))
                    if using_DiCE:
                        print("Inner Steps: {}".format(inner_steps))
                        print("Outer Steps: {}".format(outer_steps))
                    else:
                        print("Algos: {}".format(algos))
                    print("lr_policies_outer: {}".format(lr_policies_outer))
                    print("lr_policies_inner: {}".format(lr_policies_inner))
                    print("lr_values: {}".format(lr_values))
                    print("Starting Policies:")
                    if using_samples:
                        game.print_policy_and_value_info(th, vals)
                    else:
                        for i in range(n_agents):
                            policy = torch.sigmoid(th[i])
                            print("Policy {}".format(i+1))
                            print(policy)
                            if args.ill_condition:
                                print("TRANSFORMED Policy {}".format(i+1))
                                print(torch.sigmoid(ill_cond_matrices[i] @ th[i]))


                if using_samples:
                    if using_DiCE:
                        if outer_steps[0] > 1: # TODO either just use this new update loop or find a better condition to check
                            assert args.repeat_train_on_same_samples
                            th, vals = dice_update_th_new_repeat_loop(th, vals, n_agents, inner_steps, outer_steps, lr_policies_outer, lr_policies_inner, lr_values)

                        else:
                            th, vals = dice_update_th(th, vals, n_agents, inner_steps, outer_steps, lr_policies_outer, lr_policies_inner, lr_values, repeat_train_on_same_samples)

                    else:
                        # Samples but no DiCE, this is the LOLA-PG formulation
                        action_trajectory, rewards, policy_history = game.rollout(th)
                        gradient_terms = game.get_gradient_terms(action_trajectory, rewards, policy_history)

                        th, losses, G_ts, nl_terms, lola_terms, grad_2_return_1 = \
                            update_th(th, gradient_terms, lr_policies_outer, lr_policies_inner, algos, using_samples=using_samples)
                else:
                    # Exact gradient case
                    th, losses, G_ts, nl_terms, lola_terms, grad_2_return_1 = \
                        update_th(th, Ls, lr_policies_outer, lr_policies_inner, algos, using_samples=using_samples)


                if using_samples:
                    if using_DiCE:
                        # Reevaluate to get the G_ts from synchronous play
                        # (otherwise you would use values from async rollouts which
                        # usually correlates with the sync play results but is sometimes a bit weird)
                        if args.env == 'coin':
                            if repeat_train_on_same_samples:
                                raise NotImplementedError("Again repeat train not yet supported here")
                            obs_history, act_history, rewards, policy_history, val_history, \
                            next_val_history, avg_same_colour_coins_picked_total, \
                            avg_diff_colour_coins_picked_total, avg_coins_picked_total = game.rollout(
                                th, vals)

                            dice_loss, G_ts, values_loss = game.get_dice_loss(
                                rewards, policy_history, val_history,
                                next_val_history)
                        elif args.env == 'imp':
                            if repeat_train_on_same_samples:
                                raise Exception(
                                    "Repeat_train not yet supported for coin game")
                            obs_history, act_history, rewards, policy_history, \
                            val_history, next_val_history = game.rollout(
                                th, vals)

                            dice_loss, G_ts, values_loss = game.get_dice_loss(
                                act_history, rewards, policy_history,
                                val_history, next_val_history)
                        else:
                            action_trajectory, rewards, policy_history, val_history, next_val_history, obs_history = game.rollout(
                                th, vals)

                            if repeat_train_on_same_samples:
                                _, G_ts, _ = game.get_dice_loss(
                                    action_trajectory, rewards,
                                    policy_history, val_history,
                                    next_val_history,
                                    old_policy_history=policy_history, use_penalty=args.outer_penalty,
                                    use_clipping=args.outer_clip, beta=args.outer_beta)
                            else:
                                _, G_ts, _ = game.get_dice_loss(
                                    action_trajectory, rewards,
                                    policy_history, val_history,
                                    next_val_history)
                                    # use_penalty=args.outer_penalty,
                                    # use_clipping=args.outer_clip, beta=args.outer_beta)

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

                if (epoch + 1) % print_every == 0:
                    print("Epoch: " + str(epoch + 1))
                    curr = timer()
                    print("Time Elapsed: {:.1f} seconds".format(curr - start))

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
                        game.print_reward_info(G_ts,
                                                     discounted_sum_of_adjustments,
                                                     truncated_coop_payout,
                                                     inf_coop_payout, args.env)


                        # NOTE: the values may seem weird here. The reason is because
                        # each agent's reward when using LOLA-DiCE (or related algorithms)
                        # is based on their rollout/lookahead of the other agent
                        # Therefore with large enough step sizes or enough inner steps
                        # The lookahead policy may be significantly different from the actual policy
                        # And the value is based on playing the game with that lookahead policy
                        # This is the right thing to do because the values are used in the baseline for variance reduction
                        # and so we want it to match the rewards that we are using to update our policy
                        # which is based on the lookahead when we are using LOLA
                        game.print_policy_and_value_info(th, vals)


                    else:
                        for i in range(n_agents):
                            policy = torch.sigmoid(th[i])
                            print("Policy {}".format(i+1))
                            print(policy)
                            if args.ill_condition:
                                print("TRANSFORMED Policy {}".format(i+1))
                                print(torch.sigmoid(ill_cond_matrices[i] @ th[i]))

                    if args.env == 'coin':
                        print("Same Colour Coins Picked (avg over batches): {:.3f}".format(avg_same_colour_coins_picked_total))
                        print("Diff Colour Coins Picked (avg over batches): {:.3f}".format(avg_diff_colour_coins_picked_total))
                        print("Total Coins Picked (avg over batches): {:.3f}".format(avg_coins_picked_total))


            # % comparison of average individual reward to max average individual reward
            # This gives us a rough idea of how close to optimal (how close to full cooperation) we are.
            # But may not be ideal. Metrics like how often TFT is found eventually (e.g. within x epochs)
            # may be more valuable/more useful for understanding and judging.
            if using_samples:
                coop_divisor = truncated_coop_payout
            else:
                coop_divisor = inf_coop_payout
            reward_percent_of_max.append((G_ts_record.mean() + discounted_sum_of_adjustments) / coop_divisor)

            plot_results = True
            if plot_results:
                now = datetime.datetime.now()
                if using_samples:
                    avg_gts_to_plot = (G_ts_record + discounted_sum_of_adjustments).mean(dim=2).view(num_epochs, n_agents)
                else:
                    avg_gts_to_plot = G_ts_record
                plt.plot(avg_gts_to_plot)
                if using_samples:
                    plt.savefig("{}agents_outerlr{}_innerlr{}_run{}_steps{}_date{}.png".format(n_agents, args.lr_policies_outer, args.lr_policies_inner, run, "_".join(list(map(str, inner_steps))), now.strftime('%Y-%m-%d_%H-%M')))
                else:
                    plt.savefig("{}agents_outerlr{}_innerlr{}_run{}_exact_date{}.png".format(n_agents, args.lr_policies_outer, args.lr_policies_inner, run, now.strftime('%Y-%m-%d_%H-%M')))
                plt.clf()

        if args.env == 'ipd':
            print("Average reward as % of max: {:.1%}".format(
                sum(reward_percent_of_max) / repeats))
