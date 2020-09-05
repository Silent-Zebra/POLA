import math
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

from DQN import DQNAgent, NeuralNet

envs = ["harvestlite", "cleanuplite"]
env = envs[0]

num_dqn_agents = 100

altruism = True
# Yeh we can learn in this env with altruism. With game length 50 on harvest 2 agents getting 25 avg which I think is optimal
# with 0.3 regrowth ~20 optimal
# 0.2 we getting around just under 15? Needs like 300 epochs or so. And not very stable. Should be able to get around 16 optimal

# I think avg formulation is the right way to go so you avoid tweaking the lr. It still preserves signal-noise ratio

# lol we need an initialization where random action is not optimal.

def average(lst):
    return sum(lst) / len(lst)

# Episodic - then no discounting

# Part of the challenge is having multiple instantiations of these agents trained in parallel
# And they each learn/update at each point of time - it's an unstable environment we dealing with here
# What is the initial behaviour seeding? Random I guess
# Cooperate is action 0, defect is action 1 I guess (just to match the table representaton if you go 0 1 as the header)


# Let's do a cleanup like environment now
# Another class? Inheriting from a base class? Make the separate class first and we can refactor if needed



class ResourceGame:
    def __init__(self, agents, states_actions_dims, game_length:int, regrowth_rate=0.2,
                 resource_cap_multiplier=2.0):
        self.agents = agents
        self.states_actions_dims = states_actions_dims
        self.game_length = game_length
        self.turn_counter = 0
        self.regrowth_rate = regrowth_rate # way this works in this env is growth is based on regrowth_rate * currently existing num of resource available
        self.resource_cap_multiplier = resource_cap_multiplier
        self.resource_cap = self.resource_cap_multiplier * len(self.agents)
        self.resource = self.resource_cap # right now assume that we start at the resource cap

    # Episode setup
    def get_initial_state(self):
        episode_ended = False
        # print(np.zeros(1) + self.resource)
        # print(np.atleast_1d(self.resource))
        return np.atleast_1d(self.resource), episode_ended

    # def step(self, state_agent1, state_agent2, action_agent1, action_agent2):
    def step(self, state):
        # actions is a list (array?) of binary 0 or 1 actions
        # returns reward, and whether terminated
        # also updates the state
        # legal_actions = [0,1]

        actions = np.zeros(len(self.agents))
        for i in range(len(self.agents)):
            # print(state)
            action = self.agents[i].act(state, epsilon=self.agents[i].epsilon)
            actions[i] = action

        resource_consumed = sum(actions)
        # print(self.resource)
        # print(resource_consumed)

        if resource_consumed <= self.resource:
            rewards = actions # 1 reward for everyone who consumed
        else:
            rewards = actions * (self.resource / resource_consumed)

        if altruism:
            rewards = average(rewards) * np.ones_like(rewards)

        self.resource -= resource_consumed
        self.resource = max(self.resource, 0)
        self.resource *= (1 + self.regrowth_rate)
        self.resource = min(self.resource, self.resource_cap)

        self.turn_counter += 1
        done = (self.turn_counter >= self.game_length)

        new_state = np.atleast_1d(self.resource)

        return new_state, actions, rewards, done

    def train_agents(self, state, actions, rewards, new_state, episode_ended):
        for i in range(len(self.agents)):
            self.agents[i].q_learn_update(state, actions[i], rewards[i], new_state, episode_ended)


    def run_game(self, train_agents=True, verbose=False):
        init_state, _ = self.get_initial_state()
        # print(init_state)

        state = init_state
        # print(state)

        self.agent_episode_rewards = np.zeros(len(self.agents))

        done = False

        while not done:
            new_state, actions, rewards, done = self.step(state)
            # train agents
            if train_agents:
                self.train_agents(state, actions, rewards, new_state, done)
            # update reward
            self.agent_episode_rewards += rewards
            # self.agent1.reward_total += reward_agent1
            # self.agent2.reward_total += reward_agent2

            state = new_state
            if verbose:
                print(state)
                print(actions)
                print(rewards)

        if altruism:
            print(average(self.agent_episode_rewards))
        else:
            print(self.agent_episode_rewards)

        # self.agent1.episode_reward_history.append(self.agent1.episode_reward)
        # self.agent2.episode_reward_history.append(self.agent2.episode_reward)


class CleanupGame:
    def __init__(self, agents, states_actions_dims, game_length:int,
                 max_resource_growth_rate=0.5, resource_cap_multiplier=2.0,
                 waste_growth_rate=0.1, waste_cap_multiplier=2.0):
        self.agents = agents
        self.states_actions_dims = states_actions_dims
        self.game_length = game_length
        self.turn_counter = 0
        self.max_resource_growth_rate = max_resource_growth_rate # way this works is with waste full, you have 0 growth, with 0 waste,
        # you have max_growth which is growth of resource in proportion to growth_rate * num_agents
        self.resource_cap_multiplier = resource_cap_multiplier
        self.resource_cap = self.resource_cap_multiplier * len(self.agents)
        self.resource = 0.0
        self.waste_cap_multiplier = waste_cap_multiplier
        self.waste_growth_rate = waste_growth_rate # waste grows as growth_rate * num_agents at each time step
        self.waste_cap = waste_cap_multiplier * len(self.agents)
        self.waste = self.waste_cap

    # Episode setup
    def get_initial_state(self):
        episode_ended = False
        # print(np.zeros(1) + self.resource)
        # print(np.atleast_1d(self.resource))
        return np.array([self.resource, self.waste]), episode_ended

    # def step(self, state_agent1, state_agent2, action_agent1, action_agent2):
    def step(self, state):
        # actions is a list (array?) of binary 0 or 1 actions
        # returns reward, and whether terminated
        # also updates the state
        # legal_actions = [0,1]

        actions = np.zeros(len(self.agents))
        for i in range(len(self.agents)):
            # print(state)
            action = self.agents[i].act(state, epsilon=self.agents[i].epsilon)
            actions[i] = action

        resource_consumed = sum(actions)
        # print(self.resource)
        # print(resource_consumed)

        if resource_consumed <= self.resource:
            rewards = actions # 1 reward for everyone who consumed
        else:
            rewards = actions * (self.resource / resource_consumed)

        if altruism:
            rewards = average(rewards) * np.ones_like(rewards)

        self.resource -= resource_consumed
        self.resource = max(self.resource, 0)

        waste_cleaned = sum(1 - actions)

        self.waste -= waste_cleaned
        self.waste = max(self.waste, 0)

        resource_regrowth_rate = (1 - self.waste / self.waste_cap) * self.max_resource_growth_rate
        resource_growth = resource_regrowth_rate * len(self.agents) # grow in proportion to num agents

        self.resource += resource_growth
        self.resource = min(self.resource, self.resource_cap)

        self.waste += self.waste_growth_rate * self.waste_cap
        self.waste = min(self.waste, self.waste_cap)

        self.turn_counter += 1
        done = (self.turn_counter >= self.game_length)

        new_state = np.array([self.resource, self.waste])

        return new_state, actions, rewards, done

    def train_agents(self, state, actions, rewards, new_state, episode_ended):
        for i in range(len(self.agents)):
            self.agents[i].q_learn_update(state, actions[i], rewards[i], new_state, episode_ended)


    def run_game(self, train_agents=True, verbose=False):
        init_state, _ = self.get_initial_state()
        # print(init_state)

        state = init_state
        # print(state)

        self.agent_episode_rewards = np.zeros(len(self.agents))

        done = False

        while not done:
            new_state, actions, rewards, done = self.step(state)
            # train agents
            if train_agents:
                self.train_agents(state, actions, rewards, new_state, done)
            # update reward
            self.agent_episode_rewards += rewards
            # self.agent1.reward_total += reward_agent1
            # self.agent2.reward_total += reward_agent2

            if verbose:
                print("State")
                print(state)
                print("Actions")
                print(actions)
                print("Rewards")
                print(rewards)

            state = new_state

        if altruism:
            print(average(self.agent_episode_rewards))
        else:
            print(self.agent_episode_rewards)


if env == "harvestlite":
    states_actions_dims = (1, 2) # state dim then action dim
elif env == "cleanuplite":
    states_actions_dims = (2, 2)



agents = []


# for _ in range(num_rl_agents):
#     agent_pool.append(Agent(states_actions_dims))
for _ in range(num_dqn_agents):
    neural_net = NeuralNet(input_size= states_actions_dims[0],#len(states_actions_dims)-1,
                           hidden_size=8,
                           output_size=states_actions_dims[-1])
    agents.append(DQNAgent(0, 1, neural_net, lr=0.01))

epochs = 5000

for epoch in range(epochs):
    if epoch % 10 == 0:
        print("Epoch: {}".format(epoch))

    if env == "harvestlite":
        game = ResourceGame(agents, states_actions_dims, game_length=50)
    elif env == "cleanuplite":
        game = CleanupGame(agents, states_actions_dims, game_length=100)

    game.run_game()

# for agent in agents:
#     print(average(agent.episode_reward_history))




# # Uncomment for graphs
# # plot_values()
# # plot_policy_hit()
#
#
#

