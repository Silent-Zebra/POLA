import math
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

from DQN import DQNAgent, NeuralNet


def average(lst):
    return sum(lst) / len(lst)

# Episodic - then no discounting

# Part of the challenge is having multiple instantiations of these agents trained in parallel
# And they each learn/update at each point of time - it's an unstable environment we dealing with here
# What is the initial behaviour seeding? Random I guess
# Cooperate is action 0, defect is action 1 I guess (just to match the table representaton if you go 0 1 as the header)


# Initialize stuff first

# Dictionary where the keys are the (s,a) pairs
class Agent:

    # Consider optimistic initialize later

    # Consider also simple epsilon decay as the episode length increases
    # kind of like test opponent strategy first, then go into your strat

    # Of course (?) right now the RL agent doesn't learn to play optimal vs TFT
    # Because horizon is too short for it to learn that cooperation leads to later cooperation with TFT
    # Can try longer horizon (or RNN like paper suggested)
    # Or even optimistic initialization

    # TODO refactor combine duplicate code in init of DQNAgent and this agent here

    def __init__(self, states_actions_dims, epsilon=0.2, epsilon_decay=0.999, epsilon_end=0.01, alpha=0.05,
                 static_policy=False, episode_reward_history_len=500, optimistic_initialization=1000):
        # First state dim, second state dim, actions
        # Right now this should generalize to changes in the state dim (first two values)
        # but not in the action dim.
        self.states_actions_dims = states_actions_dims
        # So maybe state dim 3 is when no action has been made by either player - and we never get back to this state
        # only occurs at start of game
        self.action_values = np.zeros(self.states_actions_dims) + optimistic_initialization
        self.policy = np.zeros(self.states_actions_dims) + 0.5
        self.epsilon = epsilon # constant because dynamics constantly changing
        # In some sense epsilon is like mutations in strategy
        self.action_space_size = self.states_actions_dims[-1]
        self.reward_total = 0
        self.alpha = alpha
        self.static_policy = static_policy # when true, do not learn/update policy
        self.episode_reward_history = deque(maxlen=episode_reward_history_len)
        self.episode_reward = 0
        self.epsilon_decay = epsilon_decay
        self.epsilon_end = epsilon_end

    def epsilon_greedy_policy_update(self, s):
        # s is the old state, in tuple form
        best_action = np.argmax(self.action_values[s])
        # other_action = np.argmin(self.action_values[s])
        other_action = 1 - best_action # TODO not scalable to multiple actions

        self.policy[s][best_action] = 1 - self.epsilon + self.epsilon / self.action_space_size
        self.policy[s][other_action] = self.epsilon / self.action_space_size

        self.epsilon = max(self.epsilon_decay * self.epsilon, self.epsilon_end)

    def act(self, state, epsilon_zero=False):
        s = state
        act0_prob = self.policy[s][0]
        if epsilon_zero:
            if act0_prob >= 0.5:
                action = 0
            else:
                action = 1
        else:
            roll = random.random()
            if roll < act0_prob:
                action = 0
            else:
                action = 1
        return action


    def q_learn_update(self, state, action_taken, reward, new_state, episode_ended):
        if self.static_policy:
            return

        s = state
        a = action_taken
        r = reward
        # Find the maximum Q(s,a) across all a for the new state s':
        if episode_ended:
            max_q_sprime_a = 0
        else:
            max_q_sprime_a = np.max(self.action_values[new_state])

        self.action_values[s][a] += self.alpha * (r + max_q_sprime_a -
                                                    self.action_values[s][a])

        self.epsilon_greedy_policy_update(s)


    def uniform_random(self):
        self.policy = np.zeros(self.states_actions_dims) + 0.5
        self.static_policy = True



class ResourceGame:
    def __init__(self, agents:[Agent], states_actions_dims, game_length:int, regrowth_rate=0.5,
                 resource_cap_multiplier=2.0):
        self.agents = agents
        self.states_actions_dims = states_actions_dims
        self.game_length = game_length
        self.turn_counter = 0
        self.regrowth_rate = regrowth_rate
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
    def step(self, state, second_game_step=False):
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

        print(self.agent_episode_rewards)


        # self.agent1.episode_reward_history.append(self.agent1.episode_reward)
        # self.agent2.episode_reward_history.append(self.agent2.episode_reward)


states_actions_dims = (1, 2)
# agent1 = Agent(states_actions_dims)
# agent2 = Agent(states_actions_dims)

# agent_pool = []

# num_rl_agents = 1
num_dqn_agents = 2
# num_tft_agents = 2
# num_defect_agents = 2
# num_coop_agents = 2
# num_random_agents = 0
# num_fixed_random_agents = 0

agents = []


# for _ in range(num_rl_agents):
#     agent_pool.append(Agent(states_actions_dims))
for _ in range(num_dqn_agents):
    neural_net = NeuralNet(input_size=len(states_actions_dims)-1,
                           hidden_size=64,
                           output_size=states_actions_dims[-1])
    agents.append(DQNAgent(0, 1, neural_net))

epochs = 5000

for epoch in range(epochs):
    if epoch % 10 == 0:
        print("Epoch: {}".format(epoch))

    game = ResourceGame(agents, states_actions_dims, game_length=50)
    game.run_game()

# for agent in agents:
#     print(average(agent.episode_reward_history))




# # Uncomment for graphs
# # plot_values()
# # plot_policy_hit()
#
#
#

