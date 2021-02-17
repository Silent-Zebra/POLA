import numpy as np
import torch
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F


# x = torch.zeros((2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2))
# print(torch.zeros((2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2)))

# x = torch.rand((3,3,3))
# x = x.inverse()
# print(x)

# a = torch.FloatTensor([[[0,1],[2,3]], [[4,5],[6,7]]])
# a = torch.FloatTensor([[[4,5],[6,7]], [[0,1],[2,3]]])
# b = a.flatten()
# c = b.reshape(2,2,2)
# print(c)

# print(a.shape)
# print(a @ a)
# print((a @ a).shape)


def bin_inttensor_from_int(x, n):
    return torch.Tensor([int(d) for d in (str(bin(x))[2:]).zfill(n)])
    # return [int(d) for d in str(bin(x))[2:]]

# x=2**10+ 5
# y=bin(x)[2:]
# z=str(bin(x))[2:]
# l=[int(d) for d in str(bin(x))[2:]]
# print(bin_inttensor_from_int(x, 14))
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
    p = torch.prod(bin_mat * pc + (1-bin_mat) * pd, dim=1)
    return p

# bin_mat = (build_bin_matrix(3, 8))
# pc = torch.rand((3))
# # pc = pc.repeat(8).reshape(8, 3)
# print(bin_mat)
# print(pc)
# p = build_p_vector(3, 8, pc,  bin_mat)
#
# print(p)
# print(sum(p))
# 1/0

# def build_p_vector(n, size, pc):
#     pd = 1 - pc
#     p = torch.zeros(size)
#     for i in range(size):
#         l = bin_inttensor_from_int(i, n)
#         prob = torch.prod(l * pc + (
#                 1 - l) * pd)  # remember 1 in the list l is coop, and pd_i_0 is our defect prob
#         p[i] = prob
#     return p

def ipdn(n=2, gamma=0.96, contribution_factor=0.6, contribution_scale=False):
    dims = [2**n + 1 for _ in range(n)]
    state_space = dims[0]
    # print(dims)

    if contribution_scale:
        contribution_factor = contribution_factor * n
    else:
        assert contribution_factor > 1
    # contribution_factor = 1.7
    # contribution_factor = 0.6 * n

    bin_mat = build_bin_matrix(n, size=state_space-1)

    payout_vectors = torch.zeros((n, state_space-1)) # one vector for each player, each player has n dim vector for payouts in each of the n states
    for agent in range(n):
        for state in range(state_space-1):
            l = bin_inttensor_from_int(state, n)
            total_contrib = sum(l)
            agent_payout = total_contrib * contribution_factor / n - l[agent] # if agent contributed 1, subtract 1
            payout_vectors[agent][state] = agent_payout

    def Ls(th):

        # Theta denotes (unnormalized) action probabilities at each of the states:
        # start CC CD DC DD

        init_pc = torch.zeros(n)
        for i in range(n):
            # p_i_0 = torch.sigmoid(th[i][0:1])
            p_i_0 = torch.sigmoid(th[i][-1])
            # print(torch.sigmoid(th[i][-1]))
            init_pc[i] = p_i_0

        # Here's what we'll do for the state representation
        # binary number which increments
        # and then for 1 you can take the coop prob and 0 you can take the defect prob
        # So 111111...111 is all coop
        # and 000...000 is all defect
        # Then the last state which is 1000...000 is the start state
        # So we're kinda working backwards here... CCCC...CCC is the second last table/matrix/vector entry

        p = build_p_vector(n, state_space-1, init_pc, bin_mat)
        # p = build_p_vector(n=n, size=state_space-1, pc=init_pc)


        # TODO this part can almost certainly be optimized
        # Probabilities in the states other than the start state
        all_p_is = torch.zeros((n, state_space-1))
        for i in range(n):
            p_i = torch.sigmoid(th[i][0:-1])
            # p_i = torch.reshape(torch.sigmoid(th[i][0:-1]), (-1, 1)) # or just -1 instead of -1,1
            all_p_is[i] = p_i
        # print(all_p_is.shape)


        # TODO is there a way to optimize this part and remove the loop?
        # Transition Matrix
        # Remember now our transition matrix top left is DDD...D to DDD...D
        P = torch.zeros((state_space-1, state_space-1))
        for curr_state in range(state_space - 1):
            i = curr_state
            # pc = all_p_is[:, i, :]
            pc = all_p_is[:, i]
            p_new = build_p_vector(n, state_space-1, pc, bin_mat)
            # p_new = build_p_vector(n, state_space-1, pc)
            P[i] = p_new


        M = -torch.matmul(p, torch.inverse(torch.eye(state_space-1) - gamma * P))

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
    dims = [2, 2] # now each agent gets a vector observation, a 2 dimensional vector
    # imagine something like [1,0] instead of CD which previously was just a 1D value such as 2
    # So here we have 1 = agent last played coop, 0 = agent last played defect, 2 = start of game, no past action

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

        init_state = torch.Tensor([[2, 2]]) # repeat 2 n times, where n is num agents
        # Every agent sees same state; P1 [action, P2 action, P3 action ...]
        # State 2 is start state

        # Action prob at start of the game (higher val = more likely to coop)
        p_1_0 = th[0](init_state) # take theta 0 and pass init_state through it to get the action probs
        # Do we need sigmoid? Depends how you set up the NN
        p_2_0 = th[1](init_state)
        # p_1_0 = torch.sigmoid(th[0][0:1])
        # p_2_0 = torch.sigmoid(th[1][0:1])

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
        M = -torch.matmul(p, torch.inverse(torch.eye(4) - gamma * P))

        L_1 = torch.matmul(M, torch.reshape(payout_mat_1, (4, 1)))
        L_2 = torch.matmul(M, torch.reshape(payout_mat_2, (4, 1)))
        return [L_1, L_2]

    return dims, Ls



def ipdn_simple_state(n_agents, gamma=0.96, num_iters=50):
    n = n_agents
    # Shoot actually this doesn't work either
    # Because well you can't calculate analytically
    # To enumerate the transition matrix each probability
    # you have to do a lot of calculations
    # See https://en.wikipedia.org/wiki/Binomial_distribution#Normal_approximation
    # and the section above on Poisson binomial which is
    # what we would have here.
    # So you HAVE to do rollouts.
    # But yeah maybe let's do rollouts here before we go on to func approx

    # Contribution game
    dims = [n + 2] * n # Here this is now back to the old version where we just have
    # tabular policy. In this case the states are the number of total contributors
    # + 1 additional state for the start of the game (and another +1 because you can have
    # from 0 to n contributors
    # print(dims)

    # Each player can choose to contribute 0 or contribute 1.
    # If you contribute 0 you get reward 1 as a baseline (or avoid losing 1)
    # The contribution is multiplied by the contribution factor c and redistributed
    # evenly among all agents, including those who didn't contribute
    # In the 2 player case, c needs to be > 1.5 otherwise CC isn't better than DD
    # And c needs to be < 2 otherwise C is not dominated by D for a selfish individual
    # But as the number of agents scales, we may consider scaling the contribution factor
    contribution_factor = 1.7

    def Ls(th):


        # yeah actually I think the Ls framework doesn't work here because we cannot
        # backprop through the bernoulli sample
        # Anyway I think I should probably just rebuild from scratch, using my own framework.

        # init_state = torch.Tensor([[n+1] * n])  # repeat n+1 n times, where n is num agents
        init_state = n+1 # a simple number representation of state, where state is num cooperators,
        # or n+1 at the start of the game
        # print(init_state)
        # Every agent sees same state; P1 [action, P2 action, P3 action ...]

        state = init_state
        # print(state)

        total_sum_disc_rewards = torch.zeros(n)

        for iter in range(num_iters):
            stacked_policies = torch.stack(th)
            relevant_policies = stacked_policies[:,state]
            # print(stacked_policies)
            # print(stacked_policies[:,state])
            # print(torch.stack(th, dim=1).detach())
            # print(np.ones_like(stacked_policies.detach(), dtype=int))
            # print(np.ones_like(relevant_policies, dtype=int))
            relevant_policies = torch.sigmoid(relevant_policies)
            # print(relevant_policies)
            actions = torch.bernoulli(relevant_policies)
            # actions = np.random.binomial(np.ones_like(relevant_policies, dtype=int), relevant_policies)

            # print(actions)
            n_contribs = torch.sum(actions)
            # print(n_contribs)

            rewards = n_contribs * contribution_factor / n + (1 - actions) # because defectors keep their 1, contributors lose their 1
            discounted_rewards = rewards * gamma**iter
            total_sum_disc_rewards += discounted_rewards

            # print(rewards)

            state = int(n_contribs.item())
            # print(state)

        print(-total_sum_disc_rewards)

        return -total_sum_disc_rewards

    return dims, Ls





def ipdn_with_func_approx(n, gamma=0.96):
    # Contribution game
    dims = [n] * n # now each agent gets a vector observation, a n dimensional vector where each element is the action
    # of an agent, either 0 (defect) or 1 (coop) or 2 at the start of the game
    print(dims)

    # Each player can choose to contribute 0 or contribute 1.
    # If you contribute 0 you get reward 1 as a baseline (or avoid losing 1)
    # The contribution is multiplied by the contribution factor c and redistributed
    # evenly among all agents, including those who didn't contribute
    # In the 2 player case, c needs to be > 1.5 otherwise CC isn't better than DD
    # And c needs to be < 2 otherwise C is not dominated by D for a selfish individual
    # But as the number of agents scales, we may consider scaling the contribution factor
    # contribution_factor = 1.7
    payout_mat_1 = torch.Tensor([[-1, -3], [0, -2]])
    payout_mat_2 = torch.t(payout_mat_1)

    def Ls(th, num_iters=50):

        # In n player case we have to do stochastic rollout instead of matrix inversion
        # for expectation because the transition matrix blows up exponentially

        init_state = torch.Tensor([[2] * n]) # repeat 2 n times, where n is num agents
        print(init_state)
        # Every agent sees same state; P1 [action, P2 action, P3 action ...]

        state = init_state

        for iter in range(num_iters):

            policies = []

            for net in th:
                policy = net(state)
                policies.append(policy)

            policies = np.ndarray(policies)
            actions = np.random.binomial(np.ones_like(policies), policies)

            print(actions)

            state = torch.Tensor(actions)

            # Shit no we have to do it the other way don't we,
            # otherwise we can't backprop through the states/actions

            # get actions
            # get next state and feed to all agents

        # Only at end of episode then collect discounted sum reward for all agents and do policy updates
        # For all of the Gt's, you can sum them up and accumulate the gradient updates
        # or average them, whatever
        # Do reinforce algorithm first
        # ANd test in 2p case with naive learning first then LOLA and see what happens
        # Then n player general case
        # Also consider doing the total num cooperators formulation, you can do that in tabular
        # And see results.



        # Action prob at start of the game (higher val = more likely to coop)
        p_1_0 = th[0](init_state) # take theta 0 and pass init_state through it to get the action probs
        # Do we need sigmoid? Depends how you set up the NN
        p_2_0 = th[1](init_state)
        # p_1_0 = torch.sigmoid(th[0][0:1])
        # p_2_0 = torch.sigmoid(th[1][0:1])

        # Prob of each of the first states (after first action), CC CD DC DD
        p = torch.cat([p_1_0 * p_2_0,
                       p_1_0 * (1 - p_2_0),
                       (1 - p_1_0) * p_2_0,
                       (1 - p_1_0) * (1 - p_2_0)], dim=1)

        # Probabilities in the states other than the start state
        state_batch = torch.Tensor([[1, 1], [1, 0], [0, 1], [0, 0]])
        p_1 = th[0](state_batch)
        p_2 = th[1](state_batch)

        P = torch.cat([p_1 * p_2,
                       p_1 * (1 - p_2),
                       (1 - p_1) * p_2,
                       (1 - p_1) * (1 - p_2)], dim=1)
        M = -torch.matmul(p, torch.inverse(torch.eye(4) - gamma * P))

        L_1 = torch.matmul(M, torch.reshape(payout_mat_1, (4, 1)))
        L_2 = torch.matmul(M, torch.reshape(payout_mat_2, (4, 1)))
        return [L_1, L_2]

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
        M = -torch.matmul(p, torch.inverse(torch.eye(4) - gamma * P))

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
        # init -= logit_shift
        init[-1] += 2 * logit_shift
        init[-2] += 2 * logit_shift
        th.append(init)
    # Dims [5,5] or something, len is num agents
    # And each num represents the dim of the policy for that agent (equal to state space size with binary action/bernoulli dist)
    # for i in range(len(dims)):
        # init = torch.zeros(dims[i], requires_grad=True) - logit_shift
        # init -= logit_shift
        # init[-1] = logit_shift
        # init[-2] = logit_shift
        # th.append(init)

    return th

# TODO try again with NL too.
# Also shift the logits so all others more likely to defect, ie minus the init, but then increase the coop inits by 2x. And maybe scale the logit down to 3 or so.


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        output = F.relu(self.layer1(x))
        output = F.relu(self.layer2(output))
        output = torch.sigmoid(self.layer3(output))
        return output


def init_nn(dims):
    th = []
    # optims = []
    # Dims [2, 2] or something, len is num agents
    # And each num represents the dim of the input (state) for that agent
    for i in range(len(dims)):
        init = NeuralNet(input_size=dims[i], hidden_size=128, output_size=1)
        # for param in init.parameters():
        #     print(param)
        # 1/0
        # optimizer = torch.optim.Adam(init.parameters(), lr=lr)
        th.append(init)
        # optims.append(optimizer)
    return th #, optims

def get_gradient(function, param):
    grad = torch.autograd.grad(function, param, create_graph=True)[0]
    return grad

def get_jacobian(l, param):
    out = []
    print(l)
    for thing in l:
        print(thing)
        grad = torch.autograd.grad(thing, param, create_graph=True)[0]
        out.append(grad)
    return torch.stack(out)





def update_th(th, Ls, alpha, eta, algo, lola_terms_sum, nl_terms_sum, a=0.5, b=0.1, gam=1, ep=0.1, lss_lam=0.1, using_nn=False, beta=1):
    n = len(th)
    losses = Ls(th)

    # Compute gradients
    # This is a 2d array of all the pairwise gradient computations between all agents
    # This is useful for LOLA and the other opponent modeling stuff
    # So it is the gradient of the loss of agent j with respect to the parameters of agent i
    # When j=i this is just regular gradient update

    if using_nn:

        policies = []
        for i in range(n):
            state_batch = torch.Tensor([[0, 0], [0, 1], [1, 0], [1, 1], [2, 2]])
            policy = th[i](state_batch)
            policies.append(policy)

        # grad_L = [[[get_gradient(losses[j], param) for param in th[i].parameters()]
        #            for j in range(n)]
        #           for i in range(n)]
        grad_L = [[get_gradient(losses[j], policies[i]) for j in range(n)] for i in
                  range(n)]
    else:
        grad_L = [[get_gradient(losses[j], th[i]) for j in range(n)] for i in
              range(n)]


    if algo == 'lola':
        # So what we need for LOLA is in addition to grad[i][i] which is the regular update
        # term for oneself, we need grad[j][j] which is nabla_theta2_E[R2]
        # and the grad[j][i] is nabla_theta2_E[R1] in the LOLA paper
        # Then one you have the terms you can dot, and then take the gradient/differentiate
        # through the result.
        # Why do we dot first before taking the gradient??? Isn't that different? I might be missing something

        # TODO Can fix the seed and compare two variants to see if any different
        # afterward ask

        # test = [[(j, i) for j in range(n)] for i in range(n)]
        # print(test)
        # print(test[1][0])
        # if you wanna see why we need to flip the order
        # And for every agent i we calculate the pairwise with theta_i and theta_j for every
        # other agent j

        if using_nn:
            params_len = len(list(th[0].parameters())) # assumes all agents have same param list length

            # TODO DEC 30
            # Wait think about this
            # The output of a neural net is a policy. If you feed in the state_batch into the NN, you get a policy
            # So if you use that constructed vector as a policy, and then everything else is exactly the same, plug and play...
            # Give this a try.

            # terms = [sum([torch.sum(grad_L[j][i][k] * grad_L[j][j][k]) for k in
            #                range(params_len)
            #           for j in range(n) if j != i]) for i in range(n)]
            # grads = [
            #     [grad_L[i][i][k] - alpha * get_gradient(terms[i], param) for
            #      (k, param)
            #      in enumerate(th[i].parameters())] for i in
            #     range(n)]

            terms = [sum([torch.dot(grad_L[j][i], grad_L[j][j])
                          for j in range(n) if j != i]) for i in range(n)]

            grads = [grad_L[i][i] - alpha * eta * get_gradient(terms[i], policies[i])
                     for i in range(n)]

            # This doesn't work does it?
            # Ask for help. Ask Jakob. I've spent enough time by myself on it.
            # Regarding the implementation - have a call about it perhaps.

        else:
            terms = [sum([torch.dot(grad_L[j][i], grad_L[j][j])
                          for j in range(n) if j != i]) for i in range(n)]

            lola_terms = [alpha * eta * get_gradient(terms[i], th[i])
                     for i in range(n)]

            if lola_terms_sum is None:
                lola_terms_sum = lola_terms
            else:
                for i in range(n):
                    lola_terms_sum[i] += lola_terms[i]

            nl_terms = [grad_L[i][i]
                     for i in range(n)]

            if nl_terms_sum is None:
                nl_terms_sum = nl_terms
            else:
                for i in range(n):
                    nl_terms_sum[i] += nl_terms[i]

            # print(lola_terms)
            # grads = [grad_L[i][i] - alpha * get_gradient(terms[i], th[i])
            #          for i in range(n)]

            # grads = [grad_L[i][i] - lola_terms[i]
            #          for i in range(n)]
            grads = [nl_terms[i] - lola_terms[i] for i in range(n)]


    else:  # Naive Learning
        grads = [grad_L[i][i] for i in range(n)]

    # Update theta
    with torch.no_grad():
        for i in range(n):
            if using_nn:
                k = 0
                for param in th[i].parameters():
                    param -= alpha * grads[i][k]
                    k += 1
            else:
                th[i] -= alpha * grads[i]
    return th, losses, lola_terms_sum, nl_terms_sum



# dims, Ls = ipdn_simple_state(3)
# th = init_th(dims, std=1)
# Ls(th)
# 1/0

# TODO DEC 13
# Only create the list of bin integers once. Then use that repeatedly
# Create it not as a list but as a torch tensor
# Then instead of for loop use vectorization. This should be much faster.


theta_modes = ['tabular', 'nn']
theta_mode = 'tabular'
# theta_mode = 'nn'
assert theta_mode in theta_modes
theta_init_modes = ['standard', 'tft']
theta_init_mode = 'standard'
# theta_init_mode = 'tft'



# REPEAT some x times and store the percent coop

repeats = 5

# For each repeat/run:
num_epochs = 200

reward_percent_of_max = []

gamma=0.96
# eta = 1
# eta = 20
# eta = 10
# eta = 8
# eta = 12
# eta = 3


# n_agents = 3
contribution_factor=1.8
contribution_scale=False
# contribution_factor=0.6
# contribution_scale=True


using_nn = False

etas = [0, 1, 3, 5, 7, 10, 20] #0 should be NL

n_agents_list = [2, 3, 4, 5, 6]

for n_agents in n_agents_list:

    for eta in etas:

        for run in range(repeats):

            if theta_mode == 'tabular':
                # Initialise theta ~ Normal(0, std)
                # Select game

                dims, Ls = ipdn(n=n_agents, gamma=gamma, contribution_factor=contribution_factor, contribution_scale=contribution_scale)

                # n=4 already seems to break down... maybe it needs more epochs?
                # Also there's the question of contribution/reward scaling. With scaling it seems to work for 4... but again a bit more shaky
                # And even seems ok on 5 with scaling

                std = 1
                if theta_init_mode == 'tft':
                    # std = 0.1
                    # Basically with std higher, you're going to need higher logit shift (but only slightly, really), in order to reduce the variance
                    # and avoid random-like behaviour which could undermine the closeness/pull into the TFT basin of solutions
                    th = init_th_tft(dims, std, logit_shift=1.7)
                    # Try under NL as well:
                    # 1.7, 1.8 fails
                    # 1.85, 1.9, 2 works
                    # Under LOLA:
                    # 1, 1.5 fails
                    # 1.7 works
                    # 1.6 fails
                    # so what is the implied probability we need?
                    # 2 works
                else:
                    th = init_th(dims, std)
            elif theta_mode == 'nn':
                using_nn=True
                dims, Ls = ipd2_with_func_approx()
                th = init_nn(dims)

                #TODO Try TFT mode again with some variance...

                # th, optims = init_nn(dims)
            else:
                raise Exception()

            # Set num_epochs, learning rate and learning algo

            if using_nn:
                alpha = 0.1
                # alpha = 1
                # Wtf it actually does work with alpha=1... why does it need to be this high?
                # But not always, btw.
                # Actually it works with lower alpha like 0.2 as well. Just not always. Only sometimes.
                # beta = 10
                # With func approx it seems much more a toss up... sometimes you get coop but more often you get only 1 is coop
                # and then rarely you even get both defect
                # One TODO here is to test this more thoroughly with better gradient descent
                # methods (adam, SGD w momentum, etc) And with more epochs and diff learn rates.
            else:
                # Interesting... even tabular LOLA fails with lower alpha
                # Nvm, it can work with more epochs too with lower alpha e.g. 1000 epochs
                # Perhaps the higher lr just makes
                # t easier to escape the local optima/local equilibrium
                # And get to TFT/better equilibrium
                # alpha = 0.1
                alpha = 1
            algo = 'lola'  # ('sos', 'lola', 'la', 'sga', 'co', 'eg', 'cgd', 'lss' or 'nl')


            lola_terms_sum = None
            nl_terms_sum = None

            # Run
            losses_out = np.zeros((num_epochs, len(th)))
            # th_out = []
            for k in range(num_epochs):
                th, losses, lola_terms_sum, nl_terms_sum = update_th(th, Ls, alpha, eta, algo, lola_terms_sum, nl_terms_sum, using_nn=using_nn)
                # th_out.append([th[i].data.numpy() for i in range(len(th))])
                losses_out[k] = [loss.data.numpy() for loss in losses]


            # Below is a measure across the gradient (so first, abs value/magnitude of gradient, and then average over the number of terms in it)
            # This lets us see approximately in each state, how much influence the LOLA term has
            # (but we should also compare to the regular gradient?)
            # Doing just for agent 0 right now
            print_gradients = False
            if print_gradients:
                if lola_terms_sum is not None:
                    print("LOLA terms:")
                    print(torch.abs(lola_terms_sum[0]) / num_epochs)
                    print((lola_terms_sum[0]) / num_epochs)
                if nl_terms_sum is not None:
                    print("NL terms:")
                    print(torch.abs(nl_terms_sum[0]) / num_epochs)
                    print((nl_terms_sum[0]) / num_epochs)
                if lola_terms_sum is not None:
                    print(torch.mean(torch.abs(lola_terms_sum[0])) / num_epochs)
                    total_lola = torch.zeros(1)
                    for i in range(len(th)):
                        total_lola += torch.mean((torch.abs(lola_terms_sum[0])) / num_epochs)
                    avg_lola = total_lola / len(th)
                    print("Avg LOLA abs gradient: {0:.5f}".format(avg_lola.item()))
                if nl_terms_sum is not None:
                    print(torch.mean(torch.abs(nl_terms_sum[0])) / num_epochs)
                    total_nl = torch.zeros(1)
                    for i in range(len(th)):
                        total_nl += torch.mean((torch.abs(nl_terms_sum[0])) / num_epochs)
                    avg_nl = total_nl / len(th)
                    print("Avg NL abs gradient: {0:.5f}".format(avg_nl.item()))

            # print(losses_out.mean(axis=1))
            if not contribution_scale:
                coop_payout = 1 / (1-gamma) * (contribution_factor - 1)
            else:
                coop_payout = 1 / (1-gamma) * (contribution_factor * n_agents - 1)

            # stability_threshold = 0.95

            # Stability of cooperation (how many epochs had cooperation)
            # print((losses_out.mean(axis=1) <= stability_threshold * (-coop_payout)))
            # total_stable_coop_epochs = sum(losses_out.mean(axis=1) <= stability_threshold * (-coop_payout))
            # percent_stable_coop = total_stable_coop_epochs / num_epochs
            # print(percent_stable_coop)


            # % comparison of average individual reward to max average individual reward
            # This gives us a rough idea of how close to optimal (how close to full cooperation) we are.
            max_avg_indiv_reward = coop_payout # well actually a loss, not a reward
            reward_percent_of_max.append(-losses_out.mean() / max_avg_indiv_reward)
            # print(-losses_out.mean() / max_avg_indiv_reward)

            plot_results = False
            if plot_results:
                plt.plot(losses_out)
                plt.show()

        print("Number of agents: {}".format(n_agents))
        print("Contribution factor: {}".format(contribution_factor))
        print("Scaled contribution factor? {}".format(contribution_scale))
        print("Eta: {}".format(eta))
        print(reward_percent_of_max)
        # Average over all runs
        print("Average reward as % of max: {}".format(sum(reward_percent_of_max) / repeats))
