import numpy as np

num_agents = 1000
agent_thetas = np.zeros(num_agents) + 0.0

fix_all_but_one = True
agent_thetas[0] = 0.2

# Btw and the initialization matters then too

# Does it push all theta toward 0.5 when num agents is high because your action basically makes no difference
# ie either 0 or 1 doesn't really affect the reward
# No in fact it just leaves your thing where it was before - because no push or pull in either direction.
# Why does it push to 0.5 with higher lr?
# No it does push upwards in the right direction, just really, really slowly

print(agent_thetas)

lr = 0.0001 # Note: low lr can take long to converge. But higher lr can lead to instability/convergence toward 0 (as policy action becomes less likely it is less and less explored... but it can make a sudden jump from 0.01 to 0.99 or so if it actually makes that exploration step.

# eps = 1e-3
eps = 1e-3

train_iters = 10000
for iter in range(train_iters):

    actions = np.random.binomial(np.ones(num_agents, dtype='int32'), agent_thetas)
    # print(actions)
    # common_reward = actions.sum() / num_agents # right now a float, a single value, but we can certainly make it an array and use element wise *
    common_reward = actions.sum() # with this formulation you don't have the diminishing scale of reward problem, though relative reward (or reward:noise ratio) is still diminishing
    # yeah and then you have a problem of massive gradients because of the total reward being big, and thus drastic changes in policy
    # that's why init others to 0.0 works super well but init all others to 1.0 results in oscillation because gradient too big
    # but doesn't affect theoretical convergence as long as you throw stuff into a constant/scaling learning rate hyperparam

    # does this +/- make a difference? Basically an advantage estimate. Not really
    # common_reward = common_reward * 2 - 1

    policy_gradients = common_reward * ((actions / agent_thetas) - (1 - actions) / (1 - agent_thetas))


    if iter % 1000 == 0:
        print("Iter: {}".format(iter))
        print(common_reward)
        # print(policy_gradients)
        # print(agent_thetas)

    if fix_all_but_one:
        agent_thetas[0] += lr * policy_gradients[0]
        agent_thetas[0] = np.minimum(agent_thetas[0], 1.0 - eps)
        agent_thetas[0] = np.maximum(agent_thetas[0], 0.0 + eps)
    else:
        agent_thetas += lr * policy_gradients

        # clipping to avoid numerical instability
        agent_thetas = np.minimum(agent_thetas, 1.0 - eps)
        agent_thetas = np.maximum(agent_thetas, 0.0 + eps)


print(agent_thetas)
