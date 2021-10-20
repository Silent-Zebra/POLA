"""
Coin Game environment. Adapted from https://github.com/alshedivat/lola/blob/master/lola_dice/envs/coin_game.py
"""
import gym
import numpy as np
import torch
from LOLA_nplayers import magic_box, reverse_cumsum

# from gym.spaces import Discrete, Tuple
# from gym.spaces import prng


class CoinGameVec(gym.Env):
    """
    Vectorized Coin Game environment.
    Note: slightly deviates from the Gym API.
    """
    NUM_AGENTS = 2
    COIN_POSITIONS = NUM_AGENTS
    NUM_ACTIONS = 4
    MOVES = [
        np.array([0,  1]),
        np.array([0, -1]),
        np.array([1,  0]),
        np.array([-1, 0]),
    ]

    # def __init__(self, max_steps, batch_size, grid_size=3, num_agents=2):
    def __init__(self, max_steps, batch_size, history_len, grid_size=3, gamma=0.96):
        # self.NUM_AGENTS = num_agents
        self.n_agents = self.NUM_AGENTS
        self.max_steps = max_steps
        self.grid_size = grid_size
        self.batch_size = batch_size
        self.history_len = history_len
        # The 4 channels stand for 2 players and 2 coin positions
        self.ob_space_shape = [self.NUM_AGENTS * self.COIN_POSITIONS, grid_size, grid_size]
        self.NUM_STATES = np.prod(self.ob_space_shape)
        self.available_actions = [
            np.ones((batch_size, self.NUM_ACTIONS), dtype=int)
            for _ in range(self.NUM_AGENTS)
        ]

        self.step_count = None
        self.gamma = gamma


        self.dims_no_history = [self.NUM_AGENTS * self.COIN_POSITIONS * grid_size * grid_size] * self.NUM_AGENTS
        self.dims_with_history = [self.NUM_AGENTS * self.COIN_POSITIONS * grid_size * grid_size * self.history_len] * self.NUM_AGENTS



    def reset(self):
        self.step_count = 0
        self.red_coin = np.random.RandomState().randint(2, size=self.batch_size)
        # Agent and coin positions
        self.red_pos  = np.random.RandomState().randint(
            self.grid_size, size=(self.batch_size, 2))
        self.blue_pos = np.random.RandomState().randint(
            self.grid_size, size=(self.batch_size, 2))
        self.coin_pos = np.zeros((self.batch_size, 2), dtype=np.int8)
        for i in range(self.batch_size):
            # Make sure coins don't overlap
            while self._same_pos(self.red_pos[i], self.blue_pos[i]):
                self.blue_pos[i] = np.random.RandomState().randint(self.grid_size, size=2)
            self._generate_coin(i)
        state = self._generate_state()
        state = np.reshape(state, (self.batch_size, -1))
        observations = [state, state]
        info = [{'available_actions': aa} for aa in self.available_actions]
        return observations, info

    def _generate_coin(self, i):
        self.red_coin[i] = 1 - self.red_coin[i]
        # Make sure coin has a different position than the agent
        success = 0
        while success < 2:
            success = 0
            self.coin_pos[i] = np.random.RandomState().randint(self.grid_size, size=(2))
            success  = 1 - self._same_pos(self.red_pos[i],
                                          self.coin_pos[i])
            success += 1 - self._same_pos(self.blue_pos[i],
                                          self.coin_pos[i])

    def _same_pos(self, x, y):
        return (x == y).all()

    def _generate_state(self):
        state = np.zeros([self.batch_size] + self.ob_space_shape)
        for i in range(self.batch_size):
            state[i, 0, self.red_pos[i][0], self.red_pos[i][1]] = 1
            state[i, 1, self.blue_pos[i][0], self.blue_pos[i][1]] = 1
            if self.red_coin[i]:
                state[i, 2, self.coin_pos[i][0], self.coin_pos[i][1]] = 1
            else:
                state[i, 3, self.coin_pos[i][0], self.coin_pos[i][1]] = 1
        return state

    def step(self, actions):
        ac0, ac1 = actions

        # print(ac0)
        # print(ac1)

        self.step_count += 1

        for j in range(self.batch_size):
            a0, a1 = ac0[j], ac1[j]
            # print(a0)
            # print(a1)
            assert a0 in {0, 1, 2, 3} and a1 in {0, 1, 2, 3}

            # Move players
            self.red_pos[j] = \
                (self.red_pos[j] + self.MOVES[a0]) % self.grid_size
            self.blue_pos[j] = \
                (self.blue_pos[j] + self.MOVES[a1]) % self.grid_size

        # Compute rewards
        reward_red = np.zeros(self.batch_size)
        reward_blue = np.zeros(self.batch_size)
        for i in range(self.batch_size):
            generate = False
            if self.red_coin[i]:
                if self._same_pos(self.red_pos[i], self.coin_pos[i]):
                    generate = True
                    reward_red[i] += 1
                if self._same_pos(self.blue_pos[i], self.coin_pos[i]):
                    generate = True
                    reward_red[i] += -2
                    reward_blue[i] += 1
            else:
                if self._same_pos(self.red_pos[i], self.coin_pos[i]):
                    generate = True
                    reward_red[i] += 1
                    reward_blue[i] += -2
                if self._same_pos(self.blue_pos[i], self.coin_pos[i]):
                    generate = True
                    reward_blue[i] += 1

            if generate:
                self._generate_coin(i)

        reward = [reward_red, reward_blue]
        state = self._generate_state().reshape((self.batch_size, -1))
        observations = [state, state]
        done = (self.step_count == self.max_steps)
        info = [{'available_actions': aa} for aa in self.available_actions]

        return observations, reward, done, info



    def get_next_val_history(self, val_history, ending_state_values):
        # The notation and naming here is a bit questionable. Vals is the actual parameterized value function
        # Val_history or state_vals as in some of the other functions are the state values for the given states in
        # some rollout/trajectory

        next_val_history = torch.zeros(
            (self.max_steps, self.n_agents, self.batch_size, 1))
        next_val_history[:self.max_steps - 1, :, :, :] = \
            val_history[1:self.max_steps, :, :, :]
        next_val_history[-1, :, :, :] = ending_state_values

        return next_val_history

    def rollout(self, th, vals, gamma=0.96, full_seq_obs=True):
        # Assumes same dim obs for all players
        obs_history = torch.zeros((self.max_steps, self.n_agents, self.batch_size, self.dims_no_history[0]), dtype=torch.int)
        act_history = torch.zeros((self.max_steps, self.n_agents, self.batch_size), dtype=torch.int)
        rewards = torch.zeros(
            (self.max_steps, self.n_agents, self.batch_size))
        policy_history = torch.zeros(
            (self.max_steps, self.n_agents, self.batch_size))
        val_history = torch.zeros(
            (self.max_steps, self.n_agents, self.batch_size, 1))

        ob, info = self.reset()

        if full_seq_obs:
            # init_ob_batch = torch.zeros((self.n_agents, self.batch_size, 1, self.dims_no_history[0]))
            init_ob_batch = None

        else:
            init_ob_batch = torch.zeros((self.n_agents, self.batch_size, self.dims_with_history[0]))
            for i in range(len(ob)):
                # fill all with starting state
                start_counter = 0
                end_counter = self.dims_no_history[0]
                for j in range(self.history_len):
                    init_ob_batch[i, :, start_counter:end_counter] = torch.FloatTensor(ob[i])
                    start_counter = end_counter
                    end_counter += self.dims_no_history[0]

        ob_batch = init_ob_batch

        done = False
        # gamma_t = 1.
        iter = 0

        # Policy test
        sample_obs = torch.FloatTensor([[[0,0,0],[0,1,0],[0,0,0]],
                                  [[0, 0, 0], [1, 0, 0], [0, 0, 0]],
                                  [[0, 0, 0], [0, 0, 1], [0, 0, 0]],
                                  [[0, 0, 0], [0, 0, 0], [0, 0, 0]]]).reshape(1, 36)

        if full_seq_obs:
            sample_obs = sample_obs.reshape(1,1,36)

        for i in range(self.NUM_AGENTS):
            policy = th[i](sample_obs)
            print(policy)


        while not done:
            # obs.append(ob)
            # infos.append(info)

            policies = torch.zeros((self.n_agents, self.batch_size))
            actions = torch.zeros((self.n_agents, self.batch_size),dtype=int)

            for i in range(self.NUM_AGENTS):
                ob[i] = torch.FloatTensor(ob[i])

            if full_seq_obs:
                if ob_batch is None:
                    # print(torch.stack(ob).shape)
                    # print(torch.FloatTensor(ob))
                    # 1/0
                    # ob_batch = torch.Tensor(ob).unsqueeze(2)
                    ob_batch = torch.stack(ob).unsqueeze(2)
                    # print(ob_batch.shape)
                else:
                    # print(ob_batch)
                    # print(torch.stack(ob).unsqueeze(2).shape)
                    new_ob = torch.cat((ob_batch, torch.stack(ob).unsqueeze(2)), dim=2 )
                    # print(new_ob.shape)
                    ob_batch = new_ob
                # print(ob_batch.shape)

            else:
                if self.history_len > 1:
                    new_ob = ob_batch.clone()

                    for i in range(self.NUM_AGENTS):

                        new_ob[i, :,
                        :self.dims_no_history[0] * (self.history_len - 1)] = \
                            ob_batch[i, :, self.dims_no_history[0]:self.dims_no_history[
                                                                       0] * self.history_len]

                        new_ob[i, :,
                        self.dims_no_history[0] * (self.history_len - 1):] = ob[i]

                    ob_batch = new_ob

            for i in range(self.NUM_AGENTS):

                if full_seq_obs:

                    # print(ob_batch[i].shape)
                    # print(th[i])

                    policy = th[i](ob_batch[i])  # Should be 4-D

                    state_value = vals[i](ob_batch[i])
                    # print(state_value.shape)

                    state_value = state_value[:, -1]
                    # print(state_value.shape)



                else:

                    if self.history_len > 1:

                        policy = th[i](ob_batch[i])  # Should be 4-D
                        # print(policy.shape)

                        state_value = vals[i](ob_batch[i])
                    else:
                        policy = th[i](ob[i]) # Should be 4-D
                        # print(policy.shape)

                        state_value = vals[i](ob[i])
                        # print(state_value)

                        # print(policy)



                # TODO make sure that this works with batches
                action = torch.distributions.categorical.Categorical(probs=policy.detach()).sample()

                # print(action)


                action_prob = policy[torch.arange(0,self.batch_size,1), action]

                # action_prob = policy[:, action.unsqueeze(dim=1)]
                # action_prob = torch.gather(policy, 0, action.unsqueeze(dim=1))

                # print(action_prob)

                obs_history[iter][i] = ob[i]

                val_history[iter][i] = state_value

                policies[i] = action_prob
                # policy_history[iter][i] = action_prob
                actions[i] = action

            policy_history[iter] = policies

            # TODO update actions here with each individual action

            # Sample from categorical distribution here (th assumed to be nn that outputs 4-d softmax prob)

            act_history[iter] = actions # actions as in action of every agent

            # print("---")
            # for i in range(self.NUM_AGENTS):
            #     print(ob[i].reshape(self.batch_size, 4, 3, 3))
            # print(ob)

            # Testing Only
            # actions[-1] = torch.zeros_like(actions[-1], dtype=int)

            # print(actions)


            ob, rew, done, info = self.step(actions.numpy())

            # for i in range(self.NUM_AGENTS):
            #     print(ob[i].reshape(self.batch_size, 4, 3, 3))
            # print(rew)

            for i in range(self.NUM_AGENTS):
                rewards[iter][i] = torch.FloatTensor(rew[i])
            # acs.append(ac)
            # rews.append(rew)
            # rets.append([r * gamma_t for r in rew])
            # gamma_t *= gamma
            iter += 1

        # print(val_history)
        # 1/0
        next_val_history = self.get_next_val_history(val_history, ending_state_values=state_value) # iter doesn't even matter here as long as > 0

        return obs_history.unsqueeze(-1), act_history.unsqueeze(-1), rewards.unsqueeze(-1), policy_history.unsqueeze(-1), val_history, next_val_history
               # next_val_history

    def get_loss_helper(self, rewards, policy_history):
        num_iters = self.max_steps

        discounts = torch.cumprod(self.gamma * torch.ones((num_iters)),
                                  dim=0) / self.gamma

        # Discounted rewards, not sum of rewards which G_t is
        gamma_t_r_ts = rewards * discounts.reshape(-1, 1, 1,
                                                   1)  # implicit broadcasting done by numpy


        G_ts = reverse_cumsum(gamma_t_r_ts, dim=0)


        p_act_given_state = policy_history

        log_p_act = torch.log(p_act_given_state)
        return G_ts, gamma_t_r_ts, log_p_act, discounts



    def get_dice_loss(self, rewards, policy_history, val_history, next_val_history):

        G_ts, gamma_t_r_ts, log_p_act_or_p_act_ratio, discounts = self.get_loss_helper(
            rewards, policy_history)

        discounts = discounts.view(-1, 1, 1, 1)

        R_ts = G_ts / discounts

        # print(rewards.shape)
        # print(val_history.shape)
        # print(next_val_history.shape)

        advantages = rewards + self.gamma * next_val_history - val_history


        sum_over_agents_log_p_act_or_p_act_ratio = log_p_act_or_p_act_ratio.sum(dim=1)


        deps_up_to_t = (torch.cumsum(sum_over_agents_log_p_act_or_p_act_ratio, dim=0)).reshape(-1, 1, self.batch_size, 1)

        deps_less_than_t = deps_up_to_t - sum_over_agents_log_p_act_or_p_act_ratio.reshape(-1, 1, self.batch_size, 1) # take out the dependency in the given time step

        # Look at Loaded DiCE paper to see where this formulation comes from
        loaded_dice_rewards = ((magic_box(deps_up_to_t) - magic_box(deps_less_than_t)) * discounts * advantages).sum(dim=0).mean(dim=1)


        dice_loss = -loaded_dice_rewards

        final_state_vals = next_val_history[-1]

        # print(R_ts.shape)

        values_loss = ((R_ts + (self.gamma * discounts.flip(dims=[0])) * final_state_vals.reshape(1, *final_state_vals.shape) - val_history) ** 2).sum(dim=0).mean(dim=1)

        return dice_loss, G_ts, values_loss
