import numpy as np

learning_rules = ["policy_gradient", "q_learning"]
learning_rule = learning_rules[0]

envs = ["choose1withgradient", "simplecoop"]
env = envs[0]

# for simplecoop env
defect_reward_proportion = 0.59

num_agents = 10000 #10000

# fix_all_but_one = True
fix_all_but_one = False

def average(l):
    return (sum(l) / len(l))


if learning_rule == "policy_gradient":
    agent_thetas = np.zeros(num_agents) + 0.5

    if fix_all_but_one:
        agent_thetas[0] = 0.5

    # Btw and the initialization matters then too

    # Does it push all theta toward 0.5 when num agents is high because your action basically makes no difference
    # ie either 0 or 1 doesn't really affect the reward
    # No in fact it just leaves your thing where it was before - because no push or pull in either direction.
    # Why does it push to 0.5 with higher lr?
    # No it does push upwards in the right direction, just really, really slowly

    print(agent_thetas)

elif learning_rule == "q_learning":
    agent_q_vals = np.random.uniform(-0.5, 0.5, (num_agents, 2))  + num_agents #optimistic start for explore # Note can't just use np.zeros because argmax starts picking the first option more. This can lead to faster or slower learning




lr = 0.001 # Note: low lr can take long to converge. But higher lr can lead to instability/convergence toward 0 (as policy action becomes less likely it is less and less explored... but it can make a sudden jump from 0.01 to 0.99 or so if it actually makes that exploration step.
#  Also generally need lower lr for policy gradient

exploration_eps = 0.1
num_stable_eps = 1e-3


train_iters = 500000
for iter in range(train_iters):


    # Get actions
    if learning_rule == "policy_gradient":
        actions = np.random.binomial(np.ones(num_agents, dtype='int32'), agent_thetas)
    elif learning_rule == "q_learning":
        rand_actions = np.random.binomial(np.ones(num_agents, dtype='int32'), np.zeros(num_agents, dtype='int32') + 0.5)
        do_rand = np.random.uniform(0.0, 1.0, num_agents) < exploration_eps
        do_act = 1 - do_rand
        actions = do_act * np.argmax(agent_q_vals, axis=1) + do_rand * rand_actions
        # print(do_rand * 1)
        # actions = np.argmax(agent_q_vals, axis=1)
        # print(actions)

    # Yeh q learning is fine when you fix all the rest of the agents. Not finicky regarding the positive reward scaling.


    # Let's try now the simple coord problem where 0 is defect and gets constant 1 reward, 1 is coord and reward depends on num agents coordinating
    # or if you want to do the sum formulation... no need, we can just increase the learning rate.
    # No maybe there is a need because of numerical precision/stability issues
    # Then we can do coord = sum of num coords reward, and defect = num agents / 2 reward always. in this case defect is 50% of max coord reward.
    # We can try diff thresholds like 50%, 30% (easier), 70% (harder), in between, and see what happens as num agents increases.




    if env == "choose1withgradient":
        # print(actions)
        # common_reward = actions.sum() / num_agents # right now a float, a single value, but we can certainly make it an array and use element wise *
        common_reward = actions.sum() # with this formulation you don't have the diminishing scale of reward problem, though relative reward (or reward:noise ratio) is still diminishing
        # yeah and then you have a problem of massive gradients because of the total reward being big, and thus drastic changes in policy
        # that's why init others to 0.0 works super well but init all others to 1.0 results in oscillation because gradient too big
        # but doesn't affect theoretical convergence as long as you throw stuff into a constant/scaling learning rate hyperparam

        # does this +/- make a difference? Basically an advantage estimate. Yes it does. Can help greatly with avoiding instability for policy grad because of way policy grad works
        common_reward = common_reward * 2 - num_agents
        # Q learning seems to do better without this

        common_reward /= num_agents # average reward formulation

        rewards = common_reward * np.ones(num_agents)
    elif env == "simplecoop":
        # only cooperators get the cooperation reward here. So not a social dilemma, more of a coordination problem, the challenge
        # being you only realize cooperation is optimal when you get a bunch of agents who happen to cooperate
        num_cooperators = actions.sum()
        coop_reward = num_cooperators
        defect_reward = num_agents * defect_reward_proportion

        rewards = defect_reward * (1-actions) + coop_reward * actions # depends on action choice and num cooperators


    if learning_rule == "policy_gradient":

        policy_gradients = rewards * ((actions / agent_thetas) - (1 - actions) / (1 - agent_thetas))

        if fix_all_but_one:
            agent_thetas[0] += lr * policy_gradients[0]
            agent_thetas[0] = np.minimum(agent_thetas[0], 1.0 - num_stable_eps)
            agent_thetas[0] = np.maximum(agent_thetas[0], 0.0 + num_stable_eps)
        else:
            agent_thetas += lr * policy_gradients

            # clipping to avoid numerical instability
            agent_thetas = np.minimum(agent_thetas, 1.0 - num_stable_eps)
            agent_thetas = np.maximum(agent_thetas, 0.0 + num_stable_eps)

    elif learning_rule == "q_learning":
        # rews = reward * np.ones(num_agents)
        rews = rewards
        # print(agent_q_vals)
        # print(actions)
        # print(np.take(agent_q_vals, indices=actions))
        if fix_all_but_one:
            agent_q_vals[0, 0] += (lr * (rews - agent_q_vals[:, 0]) * (
                        1 - actions))[0]
            agent_q_vals[0, 1] += (lr * (rews - agent_q_vals[:,
                                               1]) * actions)[0]  # only update if we did the action
        else:
            agent_q_vals[:,0] += lr * (rews - agent_q_vals[:,0]) * (1-actions)
            agent_q_vals[:,1] += lr * (rews - agent_q_vals[:,1]) * actions # only update if we did the action


    if iter % 1000 == 0:
        print("Iter: {}".format(iter))
        print("Average Reward: {}".format(average(rewards)))
        print("Average Action: {}".format(average(actions)))
        # print(policy_gradients)
        if learning_rule == "policy_gradient":
            if fix_all_but_one:
                print(agent_thetas[0])
        elif learning_rule == "q_learning":
            if fix_all_but_one:
                print(agent_q_vals[0])


if learning_rule == "policy_gradient":
    if fix_all_but_one:
        print(agent_thetas[0])
    else:
        print(agent_thetas)

elif learning_rule == "q_learning":
    if fix_all_but_one:
        print(agent_q_vals[0])
    else:
        print(agent_q_vals)
