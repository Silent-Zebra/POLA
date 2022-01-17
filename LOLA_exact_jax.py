import numpy as np
import torch
import matplotlib.pyplot as plt

# import torch.nn as nn
# import torch.nn.functional as F

import datetime

import copy

import argparse

import random

from timeit import default_timer as timer

import jax.numpy as jnp
from jax import jit, vmap
import jax


# global seed

def torch_to_jnp(torch_tensor):
    return jnp.array(torch_tensor.numpy())


def bin_inttensor_from_int(x, n):
    """Converts decimal value integer x into binary representation.
    Parameter n represents the number of agents (so you fill with 0s up to the number of agents)
    Well n doesn't have to be num agents. In case of lookback (say 2 steps)
    then we may want n = 2x number of agents"""
    return jnp.array([int(d) for d in (str(bin(x))[2:]).zfill(n)])

def build_bin_matrix(n, size):
    bin_mat = jnp.zeros((size, n))
    for i in range(size):
        l = bin_inttensor_from_int(i, n)
        # bin_mat[i] = l
        bin_mat = bin_mat.at[i].set(l)
    return bin_mat

def build_p_vector(n, size, pc, bin_mat):
    # print(pc)
    # pc = jnp.expand_dims(pc, -1)
    # print(pc)
    # print(jnp.tile(pc, size))
    pc = jnp.tile(pc, size).reshape(size, n)
    pd = 1 - pc
    # print(pc)
    # print(bin_mat)
    # p = torch.zeros(size)
    p = jnp.prod(bin_mat * pc + (1 - bin_mat) * pd, axis=1)
    # print(p)
    return p

def copyNN(copy_to_net, copy_from_net):
    copy_to_net.load_state_dict(copy_from_net.state_dict())



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

        state_batch = jnp.concatenate((build_bin_matrix(dim, 2 ** dim),
                                 jnp.array([init_state_representation] * dim).reshape(1, -1)))

        # print(state_batch)

        if self.state_type == 'mnist':
            state_batch = self.build_mnist_state_from_classes(state_batch)
        elif self.state_type == 'one_hot':
            state_batch = self.build_one_hot_from_batch(state_batch,
                                                        self.action_repr_dim,
                                                                one_at_a_time=False)
        elif self.state_type == 'majorTD4':
            state_batch = self.build_one_hot_from_batch(state_batch,
                                                        self.action_repr_dim,
                                                                one_at_a_time=False,
                                                                simple_2state_build=True)

        # print(state_batch)
        # print(state_batch.shape)
        return state_batch.reshape(-1, self.action_repr_dim * self.n_agents)



    def get_nn_policy_for_batch(self, pol, state_batch):
        # state_batch = jnp.transpose(state_batch)

        # print(state_batch.shape)
        # 1/0

        if args.ill_condition:
            simple_state_repr_batch = self.one_hot_to_simple_repr(state_batch)

            simple_mask = (simple_state_repr_batch.sum(dim=-1) == 0).unsqueeze(-1)  # DD state

            policy = jax.nn.sigmoid(batched_predict()(pol, state_batch) * (self.all_state_stretch_factor) * (
                            (self.dd_stretch_factor - 1) * simple_mask + 1))

            # policy = jax.nn.sigmoid(
            #     pol.predict(state_batch) * (self.all_state_stretch_factor) * (
            #                 (self.dd_stretch_factor - 1) * simple_mask + 1))
            # quite ugly but what this simple_mask does is multiply by (dd stretch factor) in the state DD, and 1 elsewhere
            # when combined with the all_state_stretch_factor, the effect is to magnify the DD state updates (policy amplified away from 0.5),
            # and scale down the updates in other states (policy brought closer to 0.5)
        else:
            policy = jax.nn.sigmoid(batched_predict()(pol, state_batch))

        return policy

    def get_policy_for_all_states(self, th, i):
        if not args.using_nn:
            if args.ill_condition:
                policy = jax.nn.sigmoid(ill_cond_matrices[i] @ th[i])
            else:
                policy = jax.nn.sigmoid(th[i])

        else:
            state_batch = self.build_all_combs_state_batch()
            policy = self.get_nn_policy_for_batch(th[i], state_batch)
            # print(policy)
            policy = policy.squeeze(-1)
            # print(policy)
            # 1/0
            # policy = policy.reshape(-1, 1).squeeze(-1)
            # print(policy)
            # 1/0
        # print(policy)
        # print(jax.nn.sigmoid(th[i]))
        # 1/0

        return policy


    def print_policies_for_all_states(self, th):
        for i in range(len(th)):
            policy = self.get_policy_for_all_states(th, i)
            self.print_policy_info(policy, i)

    def build_one_hot_from_batch(self, curr_step_batch, one_hot_dim, one_at_a_time=True, range_end=None, simple_2state_build=False):

        # if range_end is None:
        #     range_end = self.n_agents
        curr_step_batch_one_hot = jax.nn.one_hot(
            curr_step_batch, one_hot_dim)

        # print(curr_step_batch_one_hot)

        # if simple_2state_build:
        #     new_tens = torch.cat((curr_step_batch_one_hot[:,0,:],curr_step_batch_one_hot[:,1,:]), dim=-1)
        # else:
        #     new_tens = curr_step_batch_one_hot[0]
        #     if not one_at_a_time:
        #         range_end *= self.history_len
        #
        #     for i in range(1, range_end):
        #         new_tens = jnp.concatenate((new_tens, curr_step_batch_one_hot[i]), -1)

        # curr_step_batch = new_tens.float()

        return curr_step_batch_one_hot


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
            if args.using_nn:
                self.dims = [n * history_len * self.action_repr_dim] * n
            else:
                self.dims = [2 ** n + 1] * n
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

        # For exact calculations
        self.state_space = self.dims[0]
        self.bin_mat = build_bin_matrix(self.n_agents, 2 ** self.n_agents )
        # print(self.bin_mat)
        self.payout_vectors = np.zeros((n, 2 ** self.n_agents))  # one vector for each player, state space - 1 because one is the initial state. This is the r^1 or r^2 in the LOLA paper exact gradient formulation. In the 2p case this is for DD, DC, CD, CC

        for agent in range(n):
            for state in range(2 ** self.n_agents):
                l = bin_inttensor_from_int(state, n)
                total_contrib = sum(l)
                agent_payout = total_contrib * contribution_factor / n - l[
                    agent]  # if agent contributed 1, subtract 1
                agent_payout -= adjustment_to_make_rewards_negative
                self.payout_vectors[agent][state] = agent_payout
                # self.payout_vectors = self.payout_vectors.at((agent; state)).set(agent_payout)

                # print(self.payout_vectors)

        # print(self.payout_vectors)
        self.payout_vectors = jnp.array(self.payout_vectors)
        # print(self.payout_vectors)


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




    def get_exact_loss(self, th):
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

        init_pc = jnp.zeros(self.n_agents)

        policies = []
        for i in range(self.n_agents):

            policy = self.get_policy_for_all_states(th, i)
            policies.append(policy)
            # print(policy)
            # p_i_0 = policy[-1]


            if init_state_representation == 1:
                p_i_0 = policy[-2]  # force all coop at the beginning in this special case
            else:
                p_i_0 = policy[-1] # start state is at the end; this is a (kind of arbitrary) design choice


            # print("!!!!!!!")
            # print(init_pc)
            # print(p_i_0)
            # 1/0

            init_pc = init_pc.at[i].set(p_i_0)
            # init_pc[i] = p_i_0
            # Policy represents prob of coop (taking action 1)


        p = build_p_vector(self.n_agents, 2 ** self.n_agents, init_pc, self.bin_mat)

        # This part and the below part might be optimizable (without a loop) but it doesn't seem trivial
        # Probabilities in the states other than the start state
        all_p_is = jnp.zeros((self.n_agents, 2 ** self.n_agents))
        # print(all_p_is)
        for i in range(self.n_agents):
            # print(th[i])
            # if args.ill_condition:
            #     p_i = jax.nn.sigmoid((ill_cond_matrices[i] @ th[i])[0:-1])
            # else:
            #     p_i = jax.nn.sigmoid(th[i][0:-1]

            p_i = policies[i][0:-1]

            # print(p_i)
            # print(p_i.flatten())
            # print(all_p_is[i])
            all_p_is = all_p_is.at[i].set(p_i.flatten())
            # all_p_is[i] = p_i.flatten()
            # print(all_p_is)

        # Transition Matrix
        # Remember now our transition matrix top left is DDD...D
        # 0 is defect in this formulation, 1 is contributing/cooperating
        P = jnp.zeros((2 ** self.n_agents, 2 ** self.n_agents))
        for curr_state in range(2 ** self.n_agents):
            i = curr_state
            pc = all_p_is[:, i]
            # print(pc)
            p_new = build_p_vector(self.n_agents, 2 ** self.n_agents, pc, self.bin_mat)
            # print(p_new)
            P = P.at[i].set(p_new)
            # P[i] = p_new

        # print(P)
        # print(jnp.eye(2 ** self.n_agents) - gamma * P)

        # Here instead of using infinite horizon which is the implicit assumption in the derivation from the previous parts (see again the LOLA appendix A.2)
        # We can consider a finite horizon and literally just unroll the calculation
        # You could probably just do 2 inverses and do some multiplication by discount factor
        # and subtract instead of doing this loop but oh well, this is more for testing and comparing anyway.
        if args.exact_finite_horizon:
            gamma_t_P_t = jnp.eye(2 ** self.n_agents)
            running_total = jnp.eye(2 ** self.n_agents)

            for t in range(1, args.rollout_len):
                gamma_t_P_t = gamma * gamma_t_P_t @ P
                running_total += gamma_t_P_t
            M = p @ gamma_t_P_t
        else:
            M = p @ jnp.linalg.inv(jnp.eye(2 ** self.n_agents) - gamma * P)
            # M = jnp.matmul(p, jnp.linalg.inv(
            #     jnp.eye(2 ** self.n_agents) - gamma * P))

        # Remember M is just the steady state probabilities for each of the states (discounted state visitation count, not a probability)
        # It is a vector, not a matrix.


        L_all = []
        for i in range(self.n_agents):
            payout_vec = self.payout_vectors[i]

            # print(P)
            # print(M)
            # print(payout_vec)

            # BE CAREFUL WITH SIGNS!
            # OPTIMS REQUIRE LOSSES
            rew_i = jnp.matmul(M, payout_vec)
            L_i = -rew_i
            L_all.append(L_i)

        # print(L_all)

        return L_all



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



def relu(x):
    return jnp.maximum(0, x)

def tanh(x):
    return jnp.tanh(x)

def get_random_key():
    if args.set_seed:
        raise NotImplementedError # Why is this so hard to work with
        # seed += 1
        # prngkey_num = seed
    else:
        prngkey_num = random.randint(0, 1000)
    return jax.random.PRNGKey(prngkey_num)

def get_neural_net_params(input_size, hidden_size, extra_hidden_layers, output_size):
    def random_layer_params( m, n, key, scale=args.std):
        w_key, b_key = jax.random.split(key)
        return scale * jax.random.normal(w_key, (n, m)), scale * jax.random.normal(
            b_key, (n,))

    # Initialize all layers for a fully-connected neural network with sizes "sizes"
    def init_network_params(sizes, key):
        keys = jax.random.split(key, len(sizes))
        return [random_layer_params(m, n, k) for m, n, k in
                zip(sizes[:-1], sizes[1:], keys)]



    # https://jax.readthedocs.io/en/latest/notebooks/neural_network_with_tfds_data.html
    sizes = []
    sizes.append(input_size)
    sizes.extend([hidden_size] * (extra_hidden_layers + 1))
    sizes.append(output_size)

    # if args.set_seed:
    #     prngkey_num = args.seed
    # else:
    #     prngkey_num = random.randint(0, 1000)
    #
    # self.params = self.init_network_params(sizes, random.PRNGKey(prngkey_num))

    params = init_network_params(sizes, get_random_key())

    return params

def predict(params, input):
    # per-example predictions
    activations = input
    # print(params)
    # print(th)
    for w, b in params[:-1]:
        # print(w.shape)
        # print(activations.shape)
        # print(b.shape)
        outputs = jnp.dot(w, activations) + b
        # activations = relu(outputs)
        activations = tanh(outputs)

    final_w, final_b = params[-1]
    logits = jnp.dot(final_w, activations) + final_b
    return logits

def batched_predict():
    return vmap(predict, in_axes=(None, 0))

# @jit
def update(params, grad_fn, agent_index, step_size):
    grads = grad_fn(params, agent_index)
    # print(len(params))
    # print(len(params[0]))
    # print(len(params[0][0]))
    # print(len(params[0][0][0]))
    # print("---PARAMS---")
    # for i in range(2):
    #     print("---{}----".format(i))
    #     print(params[i])
    # # print(params)
    # print("---GRADS---")
    # for i in range(2):
    #     print("---{}----".format(i))
    #     print(grads[i])
    # # print(grads)
    # 1/0
    # # print(grads)
    #
    # for i in range(len(params)):
    #
    #
    # for (w, b), (dw, db) in zip(params, grads):
    #     print("---------------")
    #     print(b)
    #     continue
    #     print("---------------")
    #     print(db)
    #     print(b - db)
    #     # TODO figure out what is actually going on with the grads
    #     # USE a smaller nn to play around with first.
    #     # Go back to the MNIST example if you need to.
    #     # print(w)
    #     1/0
    #     print(step_size)
    #     print(dw)
    #     print(step_size * dw)
    #     print(w - step_size * dw)
    #     print(b - step_size * db)
    #     # print(dw)
    #     # print(db)
    # 1/0

    return [(w - step_size * dw, b - step_size * db)
            for (w, b), (dw, db) in zip(params[agent_index], grads[agent_index])]



# TODO maybe this should go into the game definition itself and be part of that class instead of separate
def init_custom(dims, state_type, using_nn=True, env='ipd', nn_hidden_size=16, nn_extra_hidden_layers=0):
    th = []
    # f_th = []

    # NN/func approx
    if using_nn:
        for i in range(len(dims)):

            if state_type == 'mnist':
                raise NotImplementedError
                assert env == 'ipd'

                # dims[i] here is the number of agents. Because there is one MNIST
                # digit per agent's last action, thus we have an n-dimensional
                # set of MNIST images
                # conv out channels could be something other than dims[i] if you wanted.
                # policy_net = ConvFC(conv_in_channels=dims[i],
                #                     # mnist specific input is 28x28x1
                #                     conv_out_channels=dims[i],
                #                     input_size=28,
                #                     hidden_size=nn_hidden_size,
                #                     output_size=1,
                #                     final_sigmoid=False)
            else:


                policy_net = get_neural_net_params(input_size=dims[i], hidden_size=nn_hidden_size, extra_hidden_layers=nn_extra_hidden_layers,
                                  output_size=1)

            th.append(policy_net)

    # Tabular policies
    else:
        for i in range(len(dims)):
            # DONT FORGET THIS +1
            # Right now if you omit the +1 we get a bug where the first state is the prob in the all contrib state
            init = jax.random.normal(get_random_key(), (2**n_agents + 1,)) * std
            # print(init)
            th.append(init)
            # th.append(jnp.nn.init.normal_(jnp.empty(2**n_agents + 1, requires_grad=True), std=args.std))
        # 1/0

    assert len(th) == len(dims)


    return th


def get_th_copy(th):
    return copy.deepcopy(th)

def build_policy_dist(coop_probs):
    # This version just for ipdn/exact.
    defect_probs = 1 - coop_probs

    policy_dist = jnp.vstack((coop_probs, defect_probs)).t()

    # we need to do this because kl_div needs the full distribution
    # and the way we have parameterized policy here is just a coop prob
    # if you used categorical/multinomial you wouldn't have to go through this
    # The way torch kldiv works is that the first dimension is the batch, the last dimension is the probabilities.
    # The reshape just makes so that batchmean occurs over the first axis

    policy_dist = policy_dist.reshape(1, -1, 2)
    return policy_dist

def build_policy_and_target_policy_dists(policy_to_build, target_pol_to_build, i, policies_are_logits=True):
    # Note the policy and targets are individual agent ones
    # Only used in tabular case so far

    if policies_are_logits:
        if args.ill_condition:
            policy_dist = build_policy_dist(
                jax.nn.sigmoid(ill_cond_matrices[i] @ policy_to_build))

            target_policy_dist = build_policy_dist(
                jax.nn.sigmoid(ill_cond_matrices[i] @ target_pol_to_build.detach()))
        else:
            policy_dist = build_policy_dist(jax.nn.sigmoid(policy_to_build))
            target_policy_dist = build_policy_dist(
                jax.nn.sigmoid(target_pol_to_build.detach()))
    else:
        policy_dist = build_policy_dist(policy_to_build)
        target_policy_dist = build_policy_dist(target_pol_to_build.detach())
    return policy_dist, target_policy_dist


# TODO perhaps the inner_loop step can pass the lr to this prox_f and we can scale the inner lr by eta, if we want more control over it


def outer_exact_loop_step(print_info, i, new_th, static_th_copy, game, curr_pol, other_terms, curr_iter, optims_th_primes=None):
    if print_info:
        game.print_policies_for_all_states(new_th)

    outer_loss = game.get_exact_loss(new_th)

    policy = game.get_policy_for_all_states(new_th, i)
    target_policy = game.get_policy_for_all_states(static_th_copy, i)

    policy_dist, target_policy_dist = build_policy_and_target_policy_dists(
        policy, target_policy, i, policies_are_logits=False)

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
                new_th[i] -= lr_policies_outer[i] * (get_gradient(loss_i, new_th[i]))
        else:

            # print(game.get_policy_for_all_states(new_th, i))
            assert optims_th_primes is not None
            # optim_update(optims_th_primes[i], loss_i, new_th[i].parameters() )
            # print(game.get_policy_for_all_states(new_th, i))
            # 1/0


    prev_pol = curr_pol.detach().clone()
    new_pol = game.get_policy_for_all_states(new_th, i)
    curr_pol = new_pol.detach().clone()

    if print_info:
        print(kl_div)
        # game.print_policies_for_all_states(th)
        print("Prev pol")
        print(prev_pol)
        print("Curr pol")
        print(curr_pol)
        # print(jax.nn.sigmoid(curr_pol))
        # if args.ill_condition:
        #     print("Agent {} Transformed Pol".format(i + 1))
        #     print(jax.nn.sigmoid(ill_cond_matrices[i] @ curr_pol))
        print("Iter:")
        print(curr_iter)

    policy_dist, target_policy_dist = build_policy_and_target_policy_dists(
        curr_pol, prev_pol, i, policies_are_logits=False)

    curr_prev_pol_div = torch.nn.functional.kl_div(
        input=torch.log(policy_dist),
        target=target_policy_dist,
        reduction='batchmean',
        log_target=False)

    fixed_point_reached = False
    if curr_prev_pol_div < args.prox_threshold or curr_iter > args.prox_max_iters:
        print("Outer prox iters used: {}".format(curr_iter))
        if curr_iter >= args.prox_max_iters:
            print("Reached max prox iters")
        fixed_point_reached = True

    return new_th, curr_pol, fixed_point_reached


def print_exact_policy(th, i):
    # Used for exact gradient setting
    print(
        "---Agent {} Rollout---".format(i + 1))
    for j in range(len(th)):
        print("Agent {} Policy".format(j+1))
        print(jax.nn.sigmoid(th[j]))

        if args.ill_condition:
            print("Agent {} Transformed Policy".format(j + 1))
            print(jax.nn.sigmoid(ill_cond_matrices[j] @ th[j]))


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

        # original_lola_term1 = torch.dot(grad_L[1][0], grad_L[1][1])
        # original_lola_term2 = torch.dot(grad_L[0][1], grad_L[0][0])
        # original_lola_terms = [original_lola_term1, original_lola_term2]
        # original_lola_terms = [lr_policies_inner[i] * get_gradient(original_lola_terms[i], th[i])
        #     for i in range(n)]

        nl_terms = [grad_L[i][i]
                    for i in range(n)]

        print("!!!NL TERMS!!!")
        print(nl_terms)
        print("!!!LOLA TERMS!!!")
        print(lola_terms)
        # print("!!!ORIGINAL LOLA TERMS!!!")
        # print(original_lola_terms)

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


def list_dot(l1, l2):
    # assumes same length of lists
    assert len(l1) == len(l2)
    # print(l1)
    # print(l2)
    sum = 0
    for i in range(len(l1)):
        # print(torch.sum(l1[i] * l2[i]))
        sum += torch.sum(l1[i] * l2[i])
    return sum


def update_th_taylor_approx_exact_value(th, game):
    # This is the original LOLA formulation
    assert not args.no_taylor_approx
    n = len(th)

    losses = game.get_exact_loss(th)

    # Compute gradients
    # This is a 2d array of all the pairwise gradient computations between all agents
    # This is useful for LOLA and the other opponent modeling stuff
    # So it is the gradient of the loss of agent j with respect to the parameters of agent i
    # When j=i this is just regular gradient update


    if args.using_nn:
        total_params = 0
        for param in th[0].parameters():
            total_params += 1

        # print(total_params)
        # test = [[0] * total_params]
        # test[0][1] = "hi"

        grad_L = [[[0] * total_params] * n] * n
        # print(grad_L)

        # grad_L = torch.zeros((n, n, total_params))
        for i in range(n):
            for j in range(n):
                k = 0
                for param in th[i].parameters():
                    grad = get_gradient(losses[j], param)
                    # print(grad)
                    grad_L[i][j][k] = grad
                    k += 1

        # print(grad_L)
    else:
        grad_L = [[get_gradient(losses[j], th[i]) for j in range(n)] for i in
                    range(n)]

    # calculate grad_L as the gradient of loss for player j with respect to parameters for player i
    # Therefore grad_L[i][i] is simply the naive learning loss for player i

    # Be careful with mixed algorithms here; I have not tested it much

    print_info = args.print_prox_loops_info

    if args.inner_exact_prox:
        # TODO fix
        raise NotImplementedError
        static_th_copy = get_th_copy(th)

        for i in range(n):
            if args.outer_exact_prox:

                new_th = get_th_copy(static_th_copy)

                fixed_point_reached = False
                outer_iters = 0

                curr_pol = new_th[i].detach().clone()
                while not fixed_point_reached:
                    if print_info:
                        print("loop start")
                        for j in range(len(new_th)):
                            if j != i:
                                print("Agent {}".format(j))
                                print(jax.nn.sigmoid(new_th[j]))

                    new_th, other_terms = inner_exact_loop_step(new_th,
                                                                static_th_copy,
                                                                gradient_terms_or_Ls,
                                                                i, n,
                                                                prox_f_step_sizes=lr_policies_inner)

                    outer_iters += 1
                    new_th, curr_pol, fixed_point_reached = outer_exact_loop_step(
                        print_info, i, new_th, static_th_copy,
                        gradient_terms_or_Ls, curr_pol,
                        other_terms, outer_iters, args.prox_max_iters,
                        args.prox_threshold)

                th[i] = new_th[i]

            else:
                # No outer exact prox
                # This is just a single gradient step on the outer step:
                # That is, we calc the inner loop exactly
                # Use IFT to differentiate through and then get the outer gradient
                # Take 1 step, and then that's it. Move on to next loop/iteration/agent

                new_th, other_terms = inner_exact_loop_step(
                    static_th_copy, static_th_copy,
                    gradient_terms_or_Ls, i, n,
                    prox_f_step_sizes=lr_policies_inner)

                outer_rews = gradient_terms_or_Ls(new_th)

                nl_grad = get_gradient(-outer_rews[i], new_th[i])

                with torch.no_grad():
                    new_th[i] -= lr_policies_outer[i] * (
                                nl_grad + sum(other_terms))

                th[i] = new_th[i]
        return th, losses, G_ts, nl_terms, None, grad_2_return_1

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
                    lola_terms[i][k] = lr_policies_inner[i]  * -get_gradient(terms[i], param)
                    k += 1
            nl_terms = [grad_L[i][i] for i in range(n)]

            # print(nl_terms)
            # print(lola_terms)
            # 1/0

        else:

            terms = [sum([torch.dot(grad_L[j][i], grad_L[j][j])
                          for j in range(n) if j != i]) for i in range(n)]

            lola_terms = [
                lr_policies_inner[i] * -get_gradient(terms[i], th[i])
                for i in range(n)]

            nl_terms = [grad_L[i][i] for i in range(n)]

            # print("!!!NL TERMS!!!")
            # print(nl_terms)
            # if args.ill_condition:
            #     for i in range(n):
            #         print(ill_cond_matrices[i] @ nl_terms[i])
            # print("!!!LOLA TERMS!!!")
            # print(lola_terms)
            # if args.ill_condition:
            #     for i in range(n):
            #         print(ill_cond_matrices[i] @ lola_terms[i])

            grads = [nl_terms[i] + lola_terms[i] for i in range(n)]

    # Update theta
    with torch.no_grad():
        for i in range(n):
            if not isinstance(th[i], torch.Tensor):
                k = 0
                for param in th[i].parameters():
                    param -= lr_policies_outer[i] * (nl_terms[i][k] + lola_terms[i][k])
                    k += 1
            else:
                th[i] -= lr_policies_outer[i] * grads[i]

    return th



def lookahead_loss_i(th, game, i):
    assert args.no_taylor_approx
    n = len(th)

    new_th = get_th_copy(th)

    def exact_loss_wrapper(th_to_use, index_to_use):
        return game.get_exact_loss(th_to_use)[index_to_use]

    grad_fn = jax.grad(exact_loss_wrapper)

    grads = [0] * n

    if args.using_nn:
        updated_th = get_th_copy(th)

    for j in range(n):
        # Inner loop essentially
        # Each player on the copied th does a naive update (must be differentiable!)
        if j != i:
            if not args.using_nn:

                grad_L_j = grad_fn(new_th, j)
                grads[j] = grad_L_j

            else:

                updated_th_j = update(new_th, grad_fn, j, lr_policies_inner[j])
                updated_th[j] = updated_th_j
                # grad_L_j = grad_fn(new_th, j)
                # print(grad_L_j)
                # grads[j] = grad_L_j
                return game.get_exact_loss(updated_th)[i]

            # game.print_policies_for_all_states(new_th)
            # game.print_policies_for_all_states(updated_th)
            #
            # # print(new_th)
            # # print(updated_th)
            # 1/0

    for j in range(n):
        # Inner loop essentially
        # Each player on the copied th does a naive update (must be differentiable!)
        if j != i:
            if not args.using_nn:

                new_th[j] = new_th[j] - lr_policies_inner[
                    j] * grad_L_j[j]

            else:
                pass

    return game.get_exact_loss(new_th)[i]



def update_th_no_taylor_approx_exact(th, game):
    assert args.no_taylor_approx
    n = len(th)

    static_th_copy = get_th_copy(th)

    # grad_fn = jax.grad(lookahead_loss_i)

    def lookahead_loss_i_wrapper(th, i):
        return lookahead_loss_i(th, game, i)

    grad_fn = jax.grad(lookahead_loss_i_wrapper)


    for i in range(n):
        if not args.using_nn:
            lola_grad_i = grad_fn(static_th_copy, i)
            # game.print_policies_for_all_states(th)
            th[i] = th[i] - lr_policies_outer[i] * lola_grad_i[i]
            # game.print_policies_for_all_states(th)
        else:
            lola_grad_i = grad_fn(static_th_copy, i)
            updated_th_i = update(th, grad_fn, i, lr_policies_outer[i])
            th[i] = updated_th_i

    return th




def update_th_exact_value(th, game):
    raise Exception # Don't use for now
    assert args.no_taylor_approx
    # Do DiCE style rollouts except we can calculate exact Ls like follows

    # So what we will do is each player will calc losses
    # First copy the th
    # this is th'
    # well actually static_th_copy, if static,  is serving as the th
    # whereas th being updated makes it th' actually
    def exact_loss_wrapper(th_to_use, index_to_use):
        return game.get_exact_loss(th_to_use)[index_to_use]


    static_th_copy = get_th_copy(th)
    n = len(th)

    for i in range(n):
        # new_th is the theta''
        # new_th = get_th_copy(static_th_copy)

        if args.outer_exact_prox:
            fixed_point_reached = False
            outer_iters = 0

            curr_pol = game.get_policy_for_all_states(th, i).detach() # just an initialization, will be updated in the outer step loop
            # curr_pol = copy.deepcopy(th[i])
            while not fixed_point_reached:
                # treating th as th' here
                # and static_th_copy as th which isn't moving
                if args.using_nn:
                    # Reconstruct on every outer loop iter. The idea is this:
                    # Starting from the static th, take steps updating all the other player policies for x inner steps (roughly until convegence or doesn't have to be)
                    # Then update own policy, ONCE
                    # Then we repeat, starting from the static th for all other players' policies
                    # but now we have the old policies of all other players but our own updated policy
                    # Then the other players again solve the prox objective, but with our new policy
                    # And then we take another step
                    # And repeat
                    # This is analogous to what the IFT does after we have found a fixed point (though we can't warm start here)
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


                # --- INNER LOOP ---
                other_terms = None
                if args.inner_exact_prox:
                    new_th, other_terms = inner_exact_loop_step(new_th, static_th_copy,
                                          game, i, n, prox_f_step_sizes=lr_policies_inner)


                else:
                    inner_losses = game.get_exact_loss(new_th)

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

                outer_iters += 1
                if args.using_nn:
                    new_th, curr_pol, fixed_point_reached = outer_exact_loop_step(args.print_prox_loops_info, i, new_th, static_th_copy, game,
                                          curr_pol, other_terms, outer_iters,
                                          optims_th_primes)
                else:
                    new_th, curr_pol, fixed_point_reached = outer_exact_loop_step(
                        args.print_prox_loops_info, i, new_th, static_th_copy,
                        game, curr_pol, other_terms, outer_iters, None)

                if isinstance(new_th[i], torch.Tensor):
                    th[i] = new_th[i]
                else:
                    copyNN(th[i], new_th[i])


        else:
            if args.using_nn:

                new_th, optims_th_primes = \
                    construct_f_th_and_diffoptim(n_agents, i,
                                                     static_th_copy,
                                                     lr_policies_outer,
                                                     lr_policies_inner
                                                     )
            else:
                new_th = get_th_copy(static_th_copy)

            # --- INNER LOOP ---
            # Then each player calcs the losses
            other_terms = None
            if args.inner_exact_prox:
                raise NotImplementedError
                # new_th, other_terms = inner_exact_loop_step(new_th, static_th_copy,
                #                       game, i, n,
                #                       prox_f_step_sizes=lr_policies_inner)
            else:
                inner_losses = game.get_exact_loss(new_th)

                for j in range(n):
                    # Inner loop essentially
                    # Each player on the copied th does a naive update (must be differentiable!)
                    if j != i:
                        # if isinstance(new_th[j], torch.Tensor):
                        if not args.using_nn:
                            # print(game.get_exact_loss(new_th))

                            grad_fn = jax.grad(exact_loss_wrapper)
                            grad_L_j = grad_fn(new_th, j)
                            new_th[j] = new_th[j] - lr_policies_inner[
                                j] * grad_L_j[j]
                            # game.print_policies_for_all_states(new_th)

                        else:
                            optim_update(optims_th_primes[j],
                                         inner_losses[j],
                                         new_th[j].parameters())

            if args.print_inner_rollouts:
                print_exact_policy(new_th, i)

            # Then each player recalcs losses using mixed th where everyone else's is the new th but own th is the old (copied) one (do this in a for loop)
            outer_losses = game.get_exact_loss(new_th)

            if other_terms is not None:
                if isinstance(new_th[i], torch.Tensor):
                    with torch.no_grad():
                        new_th[i] -= lr_policies_outer[i] * (get_gradient(
                            outer_losses[i], new_th[i]) + sum(other_terms) )
                    # Finally we rewrite the th by copying from the created copies
                    th[i] = new_th[i]
                else:
                    # TODO Can borrow from the implementation of LOLA-PG for NN that I used before...
                    raise NotImplementedError # loook over the outer loop exact step and modify the code
                    # TODO btw we can further modularize a bunch of this code I think. Do that, and clean stuff up, while still testing it all the way
                    # And then support the inner loop as well (since Jakob is quite keen on it)
                    # Consider perhaps the ill cond experiments again too
                    # BUT TODO, first thing is to just write everything up, the critical ideas first, to get prof review

            else:
                # Finally each player updates their own (copied) th
                if not args.using_nn:
                    grad_fn = jax.grad(exact_loss_wrapper)
                    grad_L_j = grad_fn(new_th, i)
                    game.print_policies_for_all_states(new_th)
                    new_th[i] = new_th[i] - lr_policies_inner[
                        i] * grad_L_j[i]
                    game.print_policies_for_all_states(new_th)
                    # with torch.no_grad():
                    #     new_th[i] -= lr_policies_outer[i] * get_gradient(
                    #         outer_losses[i], new_th[i])
                    # # Finally we rewrite the th by copying from the created copies
                    th[i] = new_th[i]
                    1/0
                else:
                    optim_update(optims_th_primes[i], outer_losses[i], new_th[i].parameters())

                    copyNN(th[i], new_th[i])


    return th






if __name__ == "__main__":
    parser = argparse.ArgumentParser("NPLOLA")
    parser.add_argument("--env", type=str, default="ipd",
                        choices=["ipd"])
                        # choices=["ipd", "coin", "imp"]) Add these back in later
    parser.add_argument("--state_type", type=str, default="one_hot",
                        choices=['mnist', 'one_hot', 'majorTD4', 'old'],
                        help="For IPD/social dilemma, choose the state/obs representation type. One hot is the default. MNIST feeds in MNIST digits (0 or 1) instead of one hot class 0, class 1, etc. Old is there to support original/old formulation where we just had state representation 0,1,2. This is fine with tabular but causes issues with function approximation (where since 1 is coop, 2 is essentially 'super coop')")
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
    parser.add_argument("--nonlinearity", type=str, default="tanh",
                        choices=["tanh", "lrelu"])
    parser.add_argument("--nn_hidden_size", type=int, default=16)
    parser.add_argument("--nn_extra_hidden_layers", type=int, default=0)
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
    parser.add_argument("--base_cf_no_scale", type=float, default=1.6,
                        help="base contribution factor for no scaling (right now for 2 agents)")
    parser.add_argument("--base_cf_scale", type=float, default=0.6,
                        help="base contribution factor with scaling (right now for >2 agents)")
    parser.add_argument("--std", type=float, default=0.1, help="standard deviation for initialization of policy/value parameters")
    parser.add_argument("--inner_beta", type=float, default=0, help="beta determines how strong we want the KL penalty to be. Used with inner_exact_prox ")
    parser.add_argument("--outer_beta", type=float, default=0, help="beta determines how strong we want the KL penalty to be. Used with outer_exact_prox ")
    parser.add_argument("--print_inner_rollouts", action="store_true")
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
    parser.add_argument("--dd_stretch_factor", type=float, default=6., help="for ill conditioning in the func approx case, stretch logit of policy in DD state by this amount")
    parser.add_argument("--all_state_stretch_factor", type=float, default=0.33, help="for ill conditioning in the func approx case, stretch logit of policy in all states by this amount")
    parser.add_argument("--theta_init_mode", type=str, default="standard",
                        choices=['standard', 'tft'],
                        help="For IPD/social dilemma in the exact gradient/tabular setting, choose the policy initialization mode.")
    parser.add_argument("--exact_grad_calc", action="store_true",
                        help="Only calc exact gradients, don't run the algo")
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
        # seed = args.seed


    std = args.std

    # Repeats for each hyperparam setting
    # repeats = 10
    repeats = args.repeats

    if args.outer_exact_prox:
        assert args.inner_exact_prox or args.no_taylor_approx

    if args.ill_condition and not args.using_nn:

        ill_cond_matrix1 = torch.tensor([[.2, 0, 1., 0, 0.],
                                         [0., .2, 1, 0., 0.],
                                         [0, 0., 1, 0, 0.],
                                         [0, 0., 1., .2, 0.],
                                         [0, 0., 1., 0., 1.]])
        ill_cond_matrix2 = torch.tensor([[.2, 1., 0., 0, 0.],
                                         [0., 1, 0, 0., 0.],
                                         [0, 1., .2, 0, 0.],
                                         [0, 1., 0., .2, 0.],
                                         [0, 1., 0., 0., 1.]])

        ill_cond_matrix1 = torch.tensor([[3., 0, 0., 0, 0.],
                                         [0., .1, 0, 0., 0.],
                                         [0, 0., .1, 0, 0.],
                                         [0, 0., 0., .1, 0.],
                                         [0, 0., 0., 0., 1.]])
        ill_cond_matrix2 = ill_cond_matrix1

        ill_cond_matrices = torch.stack((ill_cond_matrix1, ill_cond_matrix2)) # hardcoded 2 agents for now

        print(ill_cond_matrices[0])
        print(ill_cond_matrices[1])

    # For each repeat/run:
    num_epochs = args.num_epochs
    print_every = args.print_every
    batch_size = args.batch_size
    # Bigger batch is a big part of convergence with DiCE

    gamma = args.gamma

    if args.history_len > 1:
        assert args.using_nn # Right now only supported for func approx.


    n_agents_list = args.n_agents_list

    if args.env != "ipd":
        raise NotImplementedError("No exact gradient calcs done for this env yet")


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

        lr_policies_outer = jnp.array([args.lr_policies_outer] * n_agents)
        lr_policies_inner = jnp.array([args.lr_policies_inner] * n_agents)

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

        if args.exact_grad_calc:
            # Fix later if you want to use
            1/0
            total_is_in_tft_direction_p1 = 0
            total_is_in_tft_direction_p2 = 0

            for iter in range(num_epochs):

                dims, Ls = ipdn(n=n_agents, gamma=gamma,
                                contribution_factor=contribution_factor,
                                contribution_scale=contribution_scale)
                th = init_th(dims, std=args.std)
                # th = init_th_uniform(dims)  # TODO init uniform
                # th = init_th_adversarial2(dims)
                # th = init_th_adversarial4(dims)
                # th = init_th_adversarial_coop(dims)
                for i in range(len(th)):
                    print(jax.nn.sigmoid(th[i]))
                is_in_tft_direction_p1, is_in_tft_direction_p2 = exact_grad_calc(th, Ls)
                total_is_in_tft_direction_p1 += is_in_tft_direction_p1
                total_is_in_tft_direction_p2 += is_in_tft_direction_p2
                print("% TFT direction for LOLA terms")
                print(total_is_in_tft_direction_p1 / (iter + 1))
                print(total_is_in_tft_direction_p2 / (iter + 1))
                print("P1: {}".format(total_is_in_tft_direction_p1))
                print("P2: {}".format(total_is_in_tft_direction_p2))
                print("Iters: {}".format(iter+1))

            exit()

        print("Exact Gradients")

        if args.no_taylor_approx:
            print("No Taylor Approx LOLA")
        else:
            print("Taylor Approx (Original) LOLA")

        reward_percent_of_max = []

        for run in range(repeats):

            if args.env == "imp":
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

            th = init_custom(dims, args.state_type, args.using_nn, args.env, args.nn_hidden_size, args.nn_extra_hidden_layers)

            # Run
            G_ts_record = np.zeros((num_epochs, n_agents))

            for epoch in range(num_epochs):
                if epoch == 0:
                    # print("Batch size: " + str(batch_size))

                    print("lr_policies_outer: {}".format(lr_policies_outer))
                    print("lr_policies_inner: {}".format(lr_policies_inner))
                    # print("lr_values: {}".format(lr_values))
                    print("Starting Policies:")

                    game.print_policies_for_all_states(th)

                if args.no_taylor_approx:
                    th = update_th_no_taylor_approx_exact(th, game)
                else:
                    th = update_th_taylor_approx_exact_value(th, game)

                # Reevaluate to get the G_ts from synchronous play
                losses = game.get_exact_loss(th)
                G_ts_record[epoch] = -np.stack(losses)

                if (epoch + 1) % print_every == 0:
                    print("Epoch: " + str(epoch + 1))
                    curr = timer()
                    print("Time Elapsed: {:.1f} seconds".format(curr - start))

                    print("Discounted Rewards: {}".format(G_ts_record[epoch]))
                    print("Max Avg Coop Payout (Infinite Horizon): {:.3f}".format(
                            inf_coop_payout))

                    game.print_policies_for_all_states(th)


            # % comparison of average individual reward to max average individual reward
            # This gives us a rough idea of how close to optimal (how close to full cooperation) we are.
            # But may not be ideal. Metrics like how often TFT is found eventually (e.g. within x epochs)
            # may be more valuable/more useful for understanding and judging.
            if args.exact_finite_horizon:
                print("Warning: finite horizon not well tested. May need code modification")
                coop_divisor = truncated_coop_payout
            else:
                coop_divisor = inf_coop_payout
            reward_percent_of_max.append((G_ts_record.mean() + discounted_sum_of_adjustments) / coop_divisor)

            plot_results = True
            if plot_results:
                now = datetime.datetime.now()

                avg_gts_to_plot = G_ts_record
                plt.plot(avg_gts_to_plot)

                plt.savefig("{}agents_outerlr{}_innerlr{}_run{}_exact_date{}.png".format(n_agents, args.lr_policies_outer, args.lr_policies_inner, run, now.strftime('%Y-%m-%d_%H-%M')))

                plt.clf()

        if args.env == 'ipd':
            print("Average reward as % of max: {:.1%}".format(
                sum(reward_percent_of_max) / repeats))
