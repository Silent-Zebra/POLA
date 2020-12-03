import numpy as np
import torch
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F


# Of course these updates assume we have access to the reward model.


def ipd3(gamma=0.96):
    n=3
    dims = [2**n + 1 for _ in range(n)]
    print(dims)
    payout_mat_1 = torch.Tensor([[-1, -3], [0, -2]])
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
                       (1 - p_1_0) * (1 - p_2_0)], dim=1)
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
        print(M)
        # Then dot product with the payout for each player to return the 'loss'
        # which is actually just a negative reward
        # but over the course of the entire game naturally
        L_1 = torch.matmul(M, torch.reshape(payout_mat_1, (4, 1)))
        L_2 = torch.matmul(M, torch.reshape(payout_mat_2, (4, 1)))
        return [L_1, L_2]

    return dims, Ls

def ipd2_with_func_approx(gamma=0.96):
    dims = [2, 2] # now each agent gets a vector observation, a 2 dimensional vector
    # imagine something like [1,0] instead of CD which previously was just a 1D value such as 2
    # So here we have 1 = agent last played coop, 0 = agent last played defect, 2 = start of game, no past action

    # Here's the idea: first work out func approx for the 2 player case
    # with the condensed linear state space
    # Then work out instead of analytic, an experimental solution
    # Because the analytic/inversion absolutely cannot scale to higher dims...
    # UNLESS you do a condensed state representation like just # of cooperators
    # BUt even then maybe inverting a 1000x1000 matrix takes a while... but likely still much faster than simulation
    # It's something to consider trying.

    # Instead of theta just being a tabular policy, we will use a neural net
    # as a function approximator based on the input tensor states
    # Once the 2p case done, test results to see we get similar thing with LOLA vs naive learning
    # and make sure that I set this thing up correctly
    # Before generalizing to 3, and then finally n, players
    payout_mat_1 = torch.Tensor([[-1, -3], [0, -2]])
    payout_mat_2 = torch.t(payout_mat_1)

    def Ls(th):

        # Now theta needs to be neural net parameters which we'll optimize
        # Neural net will have output dim 1 to be consistent, it just outputs a prob of coop
        # to get the normalized probability of action C or D
        # Which means theta needs to be a list of NN params (one set of params for each agent)

        init_state = torch.Tensor([[2, 2]]) # repeat 2 n times, where n is num agents
        # Every agent sees same state; P1 [action, P2 action, P3 action ...]

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
        # print(p)

        # Probabilities in the states other than the start state
        state_batch = torch.Tensor([[1, 1], [1, 0], [0, 1], [0, 0]])
        p_1 = th[0](state_batch)
        p_2 = th[1](state_batch)

        # p_1 = torch.reshape(torch.sigmoid(th[0][1:5]), (4, 1))
        # p_2 = torch.reshape(torch.sigmoid(th[1][1:5]), (4, 1))

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
    payout_mat_1 = torch.Tensor([[-1, -3], [0, -2]])
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
        init = NeuralNet(input_size=dims[i], hidden_size=16, output_size=1)
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


# TODO Dec 02 first check the math/check my work to see if I messed anything up along the way in the
# func approx
# Then afterwards ask the question I had about the ordering of when you take the grad
# Spend maybe 15-30 minutes, if cannot figure out move on to n-player case where state is total num cooperators
# No actually I think we need this.
# No it's fine, think a bit more, but then ask the question
# In particular think about whether the parameter wise NN update makes sense or not
# And otherwise try the n-player extension in tabular form with simplified state rep (num coops)

def update_th(th, Ls, alpha, algo, a=0.5, b=0.1, gam=1, ep=0.1, lss_lam=0.1, using_nn=False, beta=1):
    n = len(th)
    losses = Ls(th)

    # Compute gradients
    # This is a 2d array of all the pairwise gradient computations between all agents
    # This is useful for LOLA and the other opponent modeling stuff
    # So it is the gradient of the loss of agent j with respect to the parameters of agent i
    # When j=i this is just regular gradient update

    if using_nn:
        # stuff = [get_gradient(losses[0], param) for param in th[0].parameters()]
        # for thing in stuff:
        #     print(thing.shape)
        # 1/0
        grad_L = [[[get_gradient(losses[j], param) for param in th[i].parameters()]
                   for j in range(n)]
                  for i in range(n)]
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

            # The derivation here must be wrong
            # because each layer has to be influenced by the parameter updates not just of the same layer on the other agent
            # but of ALL the layers on the other agent...
            # THis should prob be a single value instead of 6 values or whatever
            # like if we do the individual dot prods, and add them all together
            # Then do the same thing as in the original LOLA
            # remember this is just a scalar multiplier (or is it?)
            # give it a try.
            terms = [sum([torch.sum(grad_L[j][i][k] * grad_L[j][j][k]) for k in
                           range(params_len)
                      for j in range(n) if j != i]) for i in range(n)]
            grads = [
                [grad_L[i][i][k] - alpha * get_gradient(terms[i], param) for
                 (k, param)
                 in enumerate(th[i].parameters())] for i in
                range(n)]
            # grads = [
            #     [grad_L[i][i][k] - alpha * get_gradient(terms[i][k], param) for (k, param)
            #      in enumerate(th[i].parameters())] for i in
            #     range(n)]

        else:
            terms = [sum([torch.dot(grad_L[j][i], grad_L[j][j])
                          for j in range(n) if j != i]) for i in range(n)]
            # terms = [sum([torch.sum(grad_L[j][i] * grad_L[j][j])
            #               for j in range(n) if j != i]) for i in range(n)]
            # print(terms)
            grads = [grad_L[i][i] - alpha * get_gradient(terms[i], th[i]) for i
                     in
                     range(n)]

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
    return th, losses




theta_modes = ['tabular', 'nn']
theta_mode = 'nn'
assert theta_mode in theta_modes

using_nn = False
if theta_mode == 'tabular':
    # Initialise theta ~ Normal(0, std)
    # Select game
    dims, Ls = ipd()

    std = 1
    th = init_th(dims, std)
elif theta_mode == 'nn':
    using_nn=True
    dims, Ls = ipd2_with_func_approx()
    th = init_nn(dims)

    # th, optims = init_nn(dims)
else:
    raise Exception()

# Set num_epochs, learning rate and learning algo
num_epochs = 500
if using_nn:
    alpha = 0.2
    # Wtf it actually does work with alpha=1... why does it need to be this high?
    # But not always, btw.
    # beta = 10
else:
    # Interesting... even tabular LOLA fails with lower alpha
    # Nvm, it can work with more epochs too with lower alpha e.g. 1000 epochs
    # Perhaps the higher lr just makes it easier to escape the local optima/local equilibrium
    # And get to TFT/better equilibrium
    # alpha = 0.1
    alpha = 1
algo = 'lola'  # ('sos', 'lola', 'la', 'sga', 'co', 'eg', 'cgd', 'lss' or 'nl')


# Run
losses_out = np.zeros((num_epochs, len(th)))
# th_out = []
for k in range(num_epochs):
    th, losses = update_th(th, Ls, alpha, algo, using_nn=using_nn)
    # th_out.append([th[i].data.numpy() for i in range(len(th))])
    losses_out[k] = [loss.data.numpy() for loss in losses]
plt.plot(losses_out)
plt.show()
