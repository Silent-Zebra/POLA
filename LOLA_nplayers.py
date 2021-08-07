import numpy as np
import torch
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F

init_state_representation = 2  # Change here if you want different number to represent the initial state
rollout_len = 50


def bin_inttensor_from_int(x, n):
    return torch.Tensor([int(d) for d in (str(bin(x))[2:]).zfill(n)])
    # return [int(d) for d in str(bin(x))[2:]]

def int_from_bin_inttensor(bin_tens):
    index = 0
    for i in range(len(bin_tens)):
        index += 2**i * bin_tens[-i-1]
    return int(index.item())

# bin_tens = torch.tensor([0,1,0])
# print(int_from_bin_inttensor(bin_tens))
# 1/0

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


# Of course these updates assume we have access to the reward model.

def ipd2_with_func_approx(gamma=0.96):
    dims = [2,
            2]  # now each agent gets a vector observation, a 2 dimensional vector
    # imagine something like [0,1] instead of CD which previously was just a 1D value such as 2

    # Instead of theta just being a tabular policy, we will use a neural net
    # as a function approximator based on the input tensor states
    # payout_mat_1 = torch.Tensor([[-2, -5], [0, -4]])
    payout_mat_1 = torch.Tensor([[-1, -3], [0, -2]])
    payout_mat_2 = torch.t(payout_mat_1)

    def Ls(th):
        # Now theta needs to be neural net parameters which we'll optimize
        # Neural net will have output dim 1 to be consistent, it just outputs a prob of coop
        # to get the normalized probability of action C or D
        # Which means theta needs to be a list of NN params (one set of params for each agent)

        init_state = torch.Tensor([[init_state_representation,
                                    init_state_representation]])  # repeat -1 n times, where n is num agents
        # Every agent sees same state; P1 [action, P2 action, P3 action ...]
        # State 2 is start state

        # Action prob at start of the game (higher val = more likely to coop)
        p_1_0 = th[0](
            init_state)  # take theta 0 and pass init_state through it to get the action probs
        # Do we need sigmoid? Depends how you set up the NN
        p_2_0 = th[1](init_state)

        # Prob of each of the first states (after first action), CC CD DC DD
        p = torch.cat([p_1_0 * p_2_0,
                       p_1_0 * (1 - p_2_0),
                       (1 - p_1_0) * p_2_0,
                       (1 - p_1_0) * (1 - p_2_0)], dim=1)

        # Probabilities in the states other than the start state
        # state_batch = torch.Tensor([[1, 1], [1, 0], [0, 1], [0, 0]])
        state_batch = torch.Tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
        # Let's revert this to 0 is coop and 1 is defect.
        # And 2 is start state here.
        p_1 = th[0](state_batch)
        p_2 = th[1](state_batch)

        P = torch.cat([p_1 * p_2,
                       p_1 * (1 - p_2),
                       (1 - p_1) * p_2,
                       (1 - p_1) * (1 - p_2)], dim=1)
        M = torch.matmul(p, torch.inverse(torch.eye(4) - gamma * P))

        L_1 = torch.matmul(M, torch.reshape(payout_mat_1, (4, 1)))
        L_2 = torch.matmul(M, torch.reshape(payout_mat_2, (4, 1)))
        return [L_1, L_2]

    return dims, Ls


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

        # In n player case we have to do stochastic rollout instead of matrix inversion
        # for expectation because the transition matrix blows up exponentially

        init_state = torch.Tensor([[init_state_representation] * n])  # repeat -1 n times, where n is num agents
        # print(init_state)
        # Every agent sees same state; P1 [action, P2 action, P3 action ...]

        state = init_state

        trajectory = torch.zeros((num_iters, n_agents), dtype=torch.int)
        rewards = np.zeros((num_iters, n_agents))
        policy_history = torch.zeros((num_iters, n_agents))

        discounts = np.ones(num_iters)
        for i in range(num_iters):
            discounts[i] *= gamma ** i


        # TODO: rollout can be refactored as a function
        # Lots of this code can be refactored as functions

        # First use contrib factor, keep everything exact, but make it contrib factor game.
        # Then once that working, do rollouts still with 2 players.
        # No, do rollouts first, then contrib factor. Don't want to bother with exact contrib factor formulation.

        # Below is for rollouts. We can do without it for now. First just try the basic formulation
        # with exact to see if it works.
        # Then do 2 player rollouts and see if that works.
        # Only once 2 player rollouts are working, then go to n player rollouts.
        for iter in range(num_iters):

            # policies = []
            policies = torch.zeros(n_agents)

            for i in range(n_agents):
                if isinstance(th[i], torch.Tensor):


                    if (state - init_state).sum() == 0:
                        policy = torch.sigmoid(th[i])[-1]
                    else:

                        policy = torch.sigmoid(th[i])[int_from_bin_inttensor(state)]
                else:
                    policy = th[i](state)
                # single state policy, so just a prob of coop between 0 and 1
                # print(policy)
                policies[i] = policy

            policy_history[iter] = policies

            actions = np.random.binomial(np.ones(n_agents, dtype=int),
                                         policies.detach().numpy())

            state = torch.Tensor(actions)

            trajectory[iter] = torch.Tensor(actions)

            # Note: Jul 23, 2021. I finally understand why the IPD formulation
            # in the original SOS notebook uses negative rewards everywhere
            # Policy gradient can get stuck if defect-defect is 0 reward, and
            # if your policy is always defect (or close to it), you never explore
            # and you never learn. Well, even if you explore, but only in one time step,
            # your policy won't really change much.
            # However, with negative rewards everywhere, when using PG,
            # then your current actions are always pushed towards lower probabilities
            # e.g. you will never get stuck - your action prob of whatever you last did
            # is constantly being pushed down
            # so whatever is less negative will still rise up
            # but this is much better than positive reward formulation because
            # instead of the issue where you have a positive feedback loop
            # the negative rewards will avoid a positive feedback loop
            # while at the same time continuously encouraging exploration/different policies.
            # So in defining my own rewards, I should prefer negative rewards everywhere
            # (But be careful here, negative rewards should be LESS negative for the BETTER outcomes)

            total_contrib = sum(actions)
            payout_per_agent = total_contrib * contribution_factor / n
            agent_rewards = -actions + payout_per_agent  # if agent contributed 1, subtract 1, that's what the -actions does
            agent_rewards -= adjustment_to_make_rewards_negative
            rewards[iter] = agent_rewards

            # print(actions)
            # print(agent_rewards)

        # print(rewards)

        G_ts = torch.zeros((num_iters, n_agents))
        for i in range(len(rewards)):
            G_t = torch.FloatTensor(
                (rewards[i:] * discounts[i:].reshape(-1, 1)).sum(axis=0))

            G_ts[i] = G_t

        # Discounted rewards, not sum of rewards which G_t is
        gamma_t_r_ts = rewards * discounts.reshape(-1,
                                                   1)  # implicit broadcasting done by numpy

        p_act_given_state = trajectory.float() * policy_history + (
                    1 - trajectory.float()) * (
                                        1 - policy_history)  # recall 1 is coop, so when coop action 1 taken, we look at policy which is prob coop
        # and when defect 0 is taken, we take 1-policy = prob of defect

        log_p_act = torch.log(p_act_given_state)

        # These are basically grad_i E[R_0^i] - naive learning loss
        # no LOLA loss here yet
        losses_nl = (log_p_act * G_ts).sum(
            dim=0)  # Negative because the formulation is a loss (in this codebase), not a reward
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
        # maybe to be more consistent with other term

        # For lola let's try just sum of y^t r_t
        # No I think we do need to do the two separate terms
        # And then we need to modify code around line 600 or so to take gradients
        # differently for the two different terms
        # one is grad_2 E(R_1) and the other is grad_1 grad_2 E(R_2)

        # For the first term, my own derivation showed that
        # grad_2 E(R_1) = (prop to) Sum (grad_2 (log pi_2)) G_t(1)
        # Btw you also need to think about how to generalize to n players later
        # Follow the same terms formulation
        # There's some matrix of G_ts and log probs where you can do pairwise between all players
        # And you should probably use that
        # But maybe just get 2p working first. Once 2p actually learns LOLA/TFT
        # Then scale/generalize.

        # For the grad_1 grad_2 term:
        log_p_act_sums_0_to_t = torch.zeros((num_iters, n_agents))
        # the way this will work is that the ith entry (row) in this log_p_act_sums_0_to_t
        # will be the sum of log probs from time 0 to time i
        # Then the dimension of each row is the number of agents - we have the sum of log probs
        # for each agent
        # later we will product them (but in pairwise combinations!)
        for i in range(num_iters):
            single_sum = log_p_act[:i + 1].sum(dim=0)
            # print(single_sum)
            log_p_act_sums_0_to_t[i] = single_sum

        # TODO: pairwise products instead of product of everything, which is not correct
        # TODO write out the formal math derivation (well the SOS paper already did it, no?)
        # You can sum them all up and aggregate them which can help save space maybe? But maybe not, look at the full implementation
        # including the terms calc
        # Look also at how the terms calculation is done and take inspiration from there
        # But don't try too hard to fit in, rewrite the code where necessary
        # No, try to fit it in first

        # Remember also that for p1 you want grad_1 grad_2 of R_2 (P2's return)
        # So then you also want grad_1 grad_3 of R_3
        # and so on

        grad_1_grad_2_matrix = torch.zeros((n_agents, n_agents))
        for i in range(n_agents):
            for j in range(n_agents):
                # This negative formulation of rewards is ridiculous
                # TODO refactor the codebase without any negative rewards, redo the whole thing
                # gradient ascent instead of descent
                grad_1_grad_2_matrix[i][j] = (torch.FloatTensor(gamma_t_r_ts)[:,
                                              j] * log_p_act_sums_0_to_t[:,
                                                   i] * log_p_act_sums_0_to_t[:,
                                                        j]).sum(dim=0)
        # Here entry i j is grad_i grad_j E[R_j]

        losses = losses_nl

        grad_log_p_act = []
        for i in range(n_agents):
            example_grad = get_gradient(log_p_act[0, i], th[i]) if isinstance(
                th[i], torch.Tensor) else torch.cat(
                [get_gradient(log_p_act[0, i], param).flatten() for
                 param in
                 th[i].parameters()])
            # print(example_grad)
            grad_len = len(example_grad)
            # print(grad_len  )
            grad_log_p_act.append(torch.zeros((rollout_len, grad_len)))
            # grad_log_p_act.append([0] * rollout_len)

        for i in range(n_agents):

            for t in range(rollout_len):

                grad_t = get_gradient(log_p_act[t, i], th[i]) if isinstance(
                    th[i], torch.Tensor) else torch.cat(
                    [get_gradient(log_p_act[t, i], param).flatten() for
                     param in
                     th[i].parameters()])

                # get_gradient(log_p_act_sums_0_to_t[:, j][t],
                #                                        th[j]) if isinstance(
                #    th[j], torch.Tensor) else torch.cat(
                #    [get_gradient(log_p_act_sums_0_to_t[:, j][t], param).flatten() for
                #     param in
                #     th[j].parameters()])
                # print(grad_t)

                grad_log_p_act[i][t] = grad_t

        return losses, grad_1_grad_2_matrix, log_p_times_G_t_matrix, G_ts, gamma_t_r_ts, log_p_act_sums_0_to_t, log_p_act, grad_log_p_act

        # TODO May 2021
        # you're gonna have to change the output/return format here... returning something like the actions taken instead
        # And you'll have to change the update_th code too to calculate gradients based on how changing policy changes action probability
        # And that, multiplied by the G_t as weighting, and then summed up over the episode
        # Oh and then you have to change the LOLA gradient calculation to account for this too, probably
        # Put it as a separate flag, use a REINFORCE boolean instead.
        # Later use actor critic and other architectures

        # You'll probably need a lot more episodes too with smaller learning rate
        # Maybe baseline/variance reduction too
        # Anyway all this is good practice/learning, but will prob take a while
        # Get this working for 2 players w func approx and rollouts first, before moving on to more agents and before optimizing for speed
        # And using GPUs (e.g. colab)

        # Then multiply by the gradient of the log policy

        # Only at end of episode then collect discounted sum reward for all agents and do policy updates
        # For all of the Gt's, you can sum them up and accumulate the gradient updates
        # or average them, whatever
        # Do reinforce algorithm first
        # ANd test in 2p case with naive learning first then LOLA and see what happens
        # Then n player general case
        # Also consider doing the total num cooperators formulation, you can do that in tabular
        # And see results.
        # Yeah use LOLA-PG which is basically just REINFORCE
        # We could try actor critic or even only value function (basically DQN?) methods later or separately too
        # But for now just accumulate all the trajectories and apply PG update

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


# TODO try again with NL too.
# Also shift the logits so all others more likely to defect, ie minus the init, but then increase the coop inits by 2x. And maybe scale the logit down to 3 or so.


# It's not working. It's not learning TFT. Why? What might be going wrong?
# Try rollouts with no func approx first?


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




        # self.layer1 = nn.Linear(input_size, hidden_size)
        # self.layer2 = nn.Linear(hidden_size, hidden_size)
        # self.layer3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # output = F.leaky_relu(self.layer1(x))
        # output = F.leaky_relu(self.layer2(output))
        # output = torch.sigmoid(self.layer3(output))

        output = self.net(x)

        return output


# def init_nn(dims):
#     th = []
#     # optims = []
#     # Dims [2, 2] or something, len is num agents
#     # And each num represents the dim of the input (state) for that agent
#     for i in range(len(dims)):
#         if i == 1:
#             # Test diff dimension
#             init = NeuralNet(input_size=dims[i], hidden_size=128, output_size=1)
#         else:
#             init = NeuralNet(input_size=dims[i], hidden_size=128, output_size=1)
#
#         # init = NeuralNet(input_size=dims[i], hidden_size=128, output_size=1)
#
#
#
#         # optimizer = torch.optim.Adam(init.parameters(), lr=lr)
#         th.append(init)
#         # optims.append(optimizer)
#     return th #, optims

def init_custom(dims):
    th = []

    # for i in range(len(dims)):
    #     th.append(
    #         NeuralNet(input_size=dims[i], hidden_size=16, extra_hidden_layers=0,
    #                   output_size=1))
    #
    for i in range(len(dims)):
        # DONT FORGET THIS +1
        # Right now if you omit the +1 we get a bug where the first state is the prob in the all contrib state
        th.append(torch.nn.init.normal_(torch.empty(2**n_agents + 1, requires_grad=True), std=0.1))


    # th.append(NeuralNet(input_size=dims[0], hidden_size=16, extra_hidden_layers=0,
    #               output_size=1))
    # th.append(NeuralNet(input_size=dims[0], hidden_size=16, extra_hidden_layers=0,
    #               output_size=1))
    # th.append(torch.nn.init.normal_(torch.empty(5, requires_grad=True), std=0.1))
    # th.append(torch.nn.init.normal_(torch.empty(5, requires_grad=True), std=0.1))

    # TFT init
    # logit_shift = 2
    # init = torch.zeros(5, requires_grad=True) - logit_shift
    # init[-1] += 2 * logit_shift
    # init[-2] += 2 * logit_shift
    # th.append(init)

    assert len(th) == len(dims)

    return th


def get_gradient(function, param):
    grad = torch.autograd.grad(function, param, create_graph=True)[0]
    return grad


# def get_jacobian(func, param):
#     jac = torch.autograd.functional.jacobian(func, param, create_graph=True,
#                                        strict=True, vectorize=True)
#     return jac


# def get_jacobian(l, param):
#     out = []
#     print(l)
#     for thing in l:
#         print(thing)
#         grad = torch.autograd.grad(thing, param, create_graph=True)[0]
#         out.append(grad)
#     return torch.stack(out)


def update_th(th, Ls, alphas, eta, algos, epoch,
              a=0.5, b=0.1, gam=1, ep=0.1, lss_lam=0.1, using_nn=False, beta=1):
    n = len(th)

    G_ts = None
    grad_2_return_1 = None
    nl_terms = None

    if using_nn:
        losses, grad_1_grad_2_matrix, log_p_times_G_t_matrix, G_ts, gamma_t_r_ts, log_p_act_sums_0_to_t, log_p_act, grad_log_p_act = Ls(
            th)
    else:
        losses = Ls(th)

    # Compute gradients
    # This is a 2d array of all the pairwise gradient computations between all agents
    # This is useful for LOLA and the other opponent modeling stuff
    # So it is the gradient of the loss of agent j with respect to the parameters of agent i
    # When j=i this is just regular gradient update

    if using_nn:

        state_batch = torch.cat((build_bin_matrix(n_agents, 2 ** n_agents ),
                                 torch.Tensor([
                                                  init_state_representation] * n_agents).reshape(
                                     1, -1)))



        # state_batch = torch.Tensor([[0, 0], [0, 1], [1, 0], [1, 1], [-1, -1]])
        # print(state_batch)

        policies = []
        for i in range(n):
            if isinstance(th[i], torch.Tensor):
                policy = torch.sigmoid(th[i])
            else:
                policy = th[i](state_batch)

            if epoch % print_every == 0:
                print("Policy {}".format(i))
                print(
                    "(Probabilities are for cooperation/contribution, for states 00...0 (no contrib,..., no contrib), 00...01 (only last player contrib), 00...010, 00...011, increasing in binary order ..., 11...11 , start)")

                print(policy)

                if i==0:
                    # layers = th[0].net
                    # for param in layers[0].parameters():
                    #     print(param)

                    print(
                        "Discounted Sum Rewards in this episode (removing negative adjustment): ")
                    print(G_ts[0] + discounted_sum_of_adjustments)
                    print("Max Avg Coop Payout: {:.3f}".format(coop_payout))

            policies.append(policy)

        # grad_L = [[(get_gradient(losses[j], th[i]) if isinstance(th[i], torch.Tensor) else
        #     [get_gradient(losses[j], param) for param in th[i].parameters()])
        #            for j in range(n)]
        #           for i in range(n)]




    else:
        for i in range(n):
            policy = torch.sigmoid(th[i])
            if epoch % print_every == 0:
                print("Policy {}".format(i))
                print(policy)
        grad_L = [[get_gradient(losses[j], th[i]) for j in range(n)] for i in
                  range(n)]

    # calculate grad_L as the gradient of loss for player j with respect to parameters for player i
    # Therefore grad_L[i][i] is simply the naive learning loss for player i

    # if algo == 'lola':
    if 'lola' in algos:

        # TODO Can fix the seed and compare two variants to see if any different
        # afterward ask

        # test = [[(j, i) for j in range(n)] for i in range(n)]
        # print(test)
        # print(test[1][0])
        # if you wanna see why we need to flip the order
        # And for every agent i we calculate the pairwise with theta_i and theta_j for every
        # other agent j

        if using_nn:
            # params_len = len(list(th[0].parameters())) # assumes all agents have same param list length

            # terms = [sum([torch.sum(grad_L[j][i][k] * grad_L[j][j][k]) for k in
            #                range(params_len)
            #           for j in range(n) if j != i]) for i in range(n)]
            # grads = [
            #     [grad_L[i][i][k] - alpha * eta * get_gradient(terms[i], param) for
            #      (k, param)
            #      in enumerate(th[i].parameters())] for i in
            #     range(n)]
            #

            # TODO IMPORTANT REALLY THINK ABOUT WHAT ORDER IS CORRECT HERE

            # TODO aren't only the diagonal terms here used too? E.g. when i == j
            # grad_1_grad_2_return_2 = [
            #     [get_gradient(grad_1_grad_2_matrix[i][j], th[j]) if isinstance(th[j], torch.Tensor) else
            #         torch.cat([get_gradient(grad_1_grad_2_matrix[i][j], param).flatten() for param in
            #       th[j].parameters()])
            #      for j in range(n)]
            #     for i in range(n)]

            grad_1_grad_2_return_2_new = []
            for i in range(n_agents):
                grad_1_grad_2_return_2_new.append([0] * n_agents)

            for i in range(n_agents):
                for j in range(n_agents):
                    if i != j:
                        for t in range(rollout_len):
                            # print(grad_log_p_act)
                            # print(grad_log_p_act[:t+1])

                            grad_t = torch.FloatTensor(gamma_t_r_ts)[:, j][t] * \
                                     torch.outer(
                                         grad_log_p_act[i][:t + 1].sum(dim=0),
                                         grad_log_p_act[j][:t + 1].sum(dim=0))

                            if t == 0:
                                grad_1_grad_2_return_2_new[i][j] = grad_t
                            else:
                                grad_1_grad_2_return_2_new[i][j] += grad_t

            # grad_1_grad_2_return_2 = []
            # for i in range(n_agents):
            #     grad_1_grad_2_return_2.append([0] * n_agents)
            # for i in range(n_agents):
            #     for j in range(n_agents):
            #         if i != j:
            #             for t in range(rollout_len):
            #                 grad_t = torch.FloatTensor(gamma_t_r_ts)[:,j][t] * \
            #                                                torch.outer(get_gradient(log_p_act_sums_0_to_t[:,i][t], th[i]) if isinstance(
            #                 th[i], torch.Tensor) else torch.cat([get_gradient(log_p_act_sums_0_to_t[:,i][t], param).flatten() for param in
            #                   th[i].parameters()]), get_gradient(log_p_act_sums_0_to_t[:,j][t], th[j]) if isinstance(
            #                 th[j], torch.Tensor) else torch.cat([get_gradient(log_p_act_sums_0_to_t[:,j][t], param).flatten() for param in
            #                   th[j].parameters()]))
            #
            #                 if t == 0:
            #                     grad_1_grad_2_return_2[i][j] = grad_t
            #                 else:
            #                     grad_1_grad_2_return_2[i][j] += grad_t
            # print(grad_1_grad_2_return_2_new)
            # print(grad_1_grad_2_return_2)
            # 1/0
            #
            grad_1_grad_2_return_2 = grad_1_grad_2_return_2_new

            # When we take the j i entry of the log_p_times_G_t_matrix
            # what we get is the log probs of player j and the return of player i
            # so obv we have to differentiate through w.r.t player j
            # But then entry i j of the resulting grad_2_return_1
            # is then grad_j of player i return
            # Should be ok.

            # TODO CHECK THIS IS OK. CHECK EVERYTHING.

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

            # TODO given that only the i i terms are used here (only the diagonal)
            # why do we bother calculating the whole matrix?
            # Could be a potential speedup.
            # Or alternatively reuse this matrix instead of calculating the separate grad_2_return_1 matrix

            # The reason we use alpha * eta here is because
            # alpha is the learning step of the opponent (in this case we assume same amount)
            # and then we take some fraction of that in our learning step
            # Otherwise it becomes way too large relative to our gradient
            # TODO Actually think about this, is this true? Maybe it's not true.

            # IMPORTANT NOTE: the way these grad return matrices are set up is that you should always call i j here
            # because the j i switch was done during the construction of the matrix
            # TODO we should allow different eta/learning rates across other agents
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

                            # print(k)
                            #
                            # print(grad_2_return_1[i][i][
                            #       start_pos:start_pos + param_len])
                            # print(grad_2_return_1)

                            # print(grad_2_return_1[i][i])
                            # print(lola_terms[i])
                            # print(param.size())

                            grad.append(
                                (grad_2_return_1[i][i][start_pos:start_pos + param_len] + lola_terms[i][
                                                           start_pos:start_pos + param_len]).reshape(param.size())
                            )
                            start_pos += param_len

                        # print(grad)

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
                # alphas[i] *
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
            # if using_nn:
            if not isinstance(th[i], torch.Tensor):
                k = 0
                for param in th[i].parameters():
                    param += alphas[i] * grads[i][k]
                    k += 1
            else:

                th[i] += alphas[i] * grads[i]
    return th, losses, G_ts, nl_terms, lola_terms, grad_2_return_1


# TODO DEC 13
# Only create the list of bin integers once. Then use that repeatedly
# Create it not as a list but as a torch tensor
# Then instead of for loop use vectorization. This should be much faster.


theta_modes = ['tabular', 'nn']
# theta_mode = 'tabular'
theta_mode = 'nn'
assert theta_mode in theta_modes
theta_init_modes = ['standard', 'tft']
theta_init_mode = 'standard'
# theta_init_mode = 'tft'

# Repeats for each hyperparam setting
# repeats = 10
repeats = 1

# tanh instead of relu or lrelu activation seems to help. Perhaps the gradient flow is a bit nicer that way

# For each repeat/run:
num_epochs = 8000
print_every = max(1, num_epochs / 50)
print_every = 20

gamma = 0.96

# n_agents = 3
# contribution_factor = 1.6
# contribution_scale = False
contribution_factor=0.6
contribution_scale=True


# TODO Feb 17 after can try with contr factor scale
#

using_nn = False

# Why does LOLA agent sometimes defect at start but otherwise play TFT? Policy gradient issue?
etas = [0.01 * 5]

# TODO consider making etas scale based on alphas

# etas = [0,1,2,3,4,5,6,7,8,9,10]
# etas = [0, 1, 3, 5]

# n_agents_list = [2,3,4]
n_agents_list = [10]

for n_agents in n_agents_list:

    if not contribution_scale:
        coop_payout = 1 / (1 - gamma) * (contribution_factor - 1) * \
                      (
                                  1 - gamma ** rollout_len)  # This last term here accounts for the fact that we don't go to infinity
        max_payout = 1 / (1 - gamma) * (
                    contribution_factor * (n_agents - 1) / n_agents) * \
                     (1 - gamma ** rollout_len)
    else:
        coop_payout = 1 / (1 - gamma) * (contribution_factor * n_agents - 1) * \
                      (1 - gamma ** rollout_len)
        max_payout = 1 / (1 - gamma) * (contribution_factor * (n_agents - 1)) * \
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

        reward_percent_of_max = []

        for run in range(repeats):

            if theta_mode == 'tabular':

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
            elif theta_mode == 'nn':
                using_nn = True

                dims, Ls = contrib_game_with_func_approx(n=n_agents, gamma=gamma,
                                                         contribution_factor=contribution_factor,
                                                         contribution_scale=contribution_scale)


                th = init_custom(dims)

            else:
                raise Exception()


            if using_nn:
                # alphas = [0.01, 0.005]

                # alphas = [0.001, 0.001, 0.001]
                alphas = [0.01] * n_agents

                # I think part of the issue is if policy saturates at cooperation it never explores and never tries defect
                # How does standard reinforce/policy gradient get around this? Entropy regularization
                # Baseline might help too. As might negative rewards everywhere.

            else:

                alphas = [1,1]
                alphas = [0.05,0.05]
            # algos = ['nl', 'lola']
            # algos = ['lola', 'nl']
            algos = ['lola'] * n_agents


            # Run
            losses_out = np.zeros((num_epochs, n_agents))
            G_ts_record = np.zeros((num_epochs, n_agents))
            lola_terms_running_total = []
            nl_terms_running_total = []


            # th_out = []
            for k in range(num_epochs):

                # print(update_th(th, Ls, alphas, eta, algos, lola_terms_sum, nl_terms_sum, using_nn=using_nn, epoch=k))

                # th, losses, lola_terms_sum, nl_terms_sum, G_ts, lola_terms, grad_2_return_1 = \
                th, losses, G_ts, nl_terms, lola_terms, grad_2_return_1 = \
                    update_th(th, Ls, alphas, eta, algos, using_nn=using_nn, epoch=k)

                losses_out[k] = [loss.data.numpy() for loss in losses]

                if G_ts is not None:
                    G_ts_record[k] = G_ts[0]

                # This is just to get an idea of the relative influence of the lola vs nl terms
                # if lola_terms_running_total == []:
                #     for i in range(n_agents):
                #         lola_terms_running_total.append(alphas[i] * lola_terms[i].detach())
                # else:
                #     for i in range(n_agents):
                #         lola_terms_running_total[i] += alphas[i] * lola_terms[i].detach()
                #
                # for i in range(n_agents):
                #     if grad_2_return_1 is None:
                #         nl_term = nl_terms[i].detach()
                #     else:
                #         nl_term = grad_2_return_1[i][i].detach()
                #
                #     if len(nl_terms_running_total) < n_agents:
                #         nl_terms_running_total.append(alphas[i] * nl_term)
                #     else:
                #         nl_terms_running_total[i] += alphas[i] * nl_term



                if k % print_every == 0:
                    print("Epoch: " + str(k))
                    print("Eta: " + str(eta))
                    print("Algos: {}".format(algos))
                    print("Alphas: {}".format(alphas))
                    # print("LOLA Terms: ")
                    # print(lola_terms_running_total)
                    # print("NL Terms: ")
                    # print(nl_terms_running_total)

            # % comparison of average individual reward to max average individual reward
            # This gives us a rough idea of how close to optimal (how close to full cooperation) we are.
            reward_percent_of_max.append((G_ts_record.mean() + discounted_sum_of_adjustments) / coop_payout)

            plot_results = True
            if plot_results:
                plt.plot(G_ts_record + discounted_sum_of_adjustments)

                plt.show()

        print("Number of agents: {}".format(n_agents))
        print("Contribution factor: {}".format(contribution_factor))
        print("Scaled contribution factor? {}".format(contribution_scale))
        print("Eta: {}".format(eta))
        # print(reward_percent_of_max)
        # Average over all runs
        print("Average reward as % of max: {:.1%}".format(
            sum(reward_percent_of_max) / repeats))
