"""
adapted from https://github.com/alshedivat/lola/blob/master/lola_dice/envs/matching_pennies.py

Matching pennies environment.
"""
import gym
import numpy as np
import torch

from gym.spaces import Discrete, Tuple

# from .common import OneHot

from LOLA_nplayers import magic_box, reverse_cumsum, Game

class IteratedMatchingPennies(gym.Env, Game):
    """
    A two-agent environment for the Matching Pennies game.
    """
    NUM_AGENTS = 2
    NUM_ACTIONS = 2
    NUM_STATES = 5

    def __init__(self, n, batch_size, num_iters, gamma=0.9, history_len=1, state_type='one_hot'):
        self.max_steps = num_iters
        self.batch_size = batch_size
        self.gamma = gamma
        # self.payout_mat = np.array([[1, -1],[-1, 1]])
        self.payout_mat = torch.tensor([[1, -1],[-1, 1]])

        self.action_space = Tuple([
            Discrete(self.NUM_ACTIONS) for _ in range(self.NUM_AGENTS)
        ])
        # self.observation_space = Tuple([
        #     OneHot(self.NUM_STATES) for _ in range(self.NUM_AGENTS)
        # ])
        self.n_agents = n
        self.history_len = history_len
        self.available_actions = [
            np.ones((batch_size, self.NUM_ACTIONS), dtype=int)
            for _ in range(self.NUM_AGENTS)
        ]
        self.state_type = state_type
        if self.state_type == 'one_hot':
            self.action_repr_dim = 3
        else:
            self.action_repr_dim = 1
        self.step_count = None
        # self.dims = [n * history_len * self.action_repr_dim] * n
        self.dims = [self.NUM_STATES] * n



    def reset(self):
        self.step_count = 0
        init_state = np.zeros((self.batch_size, self.NUM_STATES))
        init_state[:, -1] = 1
        observations = [init_state, init_state]
        info = [{'available_actions': aa} for aa in self.available_actions]
        return observations, info

    def step(self, action):
        ac0, ac1 = action

        self.step_count += 1

        rewards = []
        state0 = np.zeros((self.batch_size, self.NUM_STATES))
        # state1 = np.zeros((self.batch_size, self.NUM_STATES))
        for i, (a0, a1) in enumerate(zip(ac0, ac1)):

            # print(self.payout_mat)
            # print(a0, a1)
            # print(self.payout_mat[a1])
            # print("---")
            #
            # print(self.payout_mat[a0, a1])
            # print(-self.payout_mat[a0, a1])

            rewards.append([self.payout_mat[a0, a1], -self.payout_mat[a0, a1]])
            state0[i, a0 * 2 + a1] = 1
            # state1[i, a1 * 2 + a0] = 1
        rewards = list(map(np.asarray, zip(*rewards)))
        # observations = [state0, state1]
        observations = [state0, state0] # Why  not just same state for all agents? This would be consistent with IPD

        # print(observations)
        # 1/0

        done = (self.step_count == self.max_steps)
        info = [{'available_actions': aa} for aa in self.available_actions]

        return observations, rewards, done, info


    def rollout(self, th, vals):

        # init_obs, info = self.reset()
        #
        # state_batch = init_obs

        # obs_history = torch.zeros((self.num_iters, self.batch_size, self.n_agents * self.action_repr_dim * self.history_len))
        # # act_history just tracks actions, doesn't track the init state
        # action_act_history = torch.zeros((self.num_iters, self.n_agents, self.batch_size, 1), dtype=torch.int)
        # rewards = torch.zeros((self.num_iters, self.n_agents, self.batch_size, 1))
        # policy_history = torch.zeros((self.num_iters, self.n_agents, self.batch_size, 1))
        # val_history = torch.zeros((self.num_iters, self.n_agents, self.batch_size, 1))

        obs_history = torch.zeros((self.max_steps, self.n_agents,
                                   self.batch_size, self.dims[0]),
                                  dtype=torch.int)
        act_history = torch.zeros(
            (self.max_steps, self.n_agents, self.batch_size, 1), dtype=torch.int)
        rewards = torch.zeros(
            (self.max_steps, self.n_agents, self.batch_size))
        policy_history = torch.zeros(
            (self.max_steps, self.n_agents, self.batch_size, 1))
        val_history = torch.zeros(
            (self.max_steps, self.n_agents, self.batch_size, 1))

        ob, info = self.reset()

        init_ob_batch = ob
        # print(ob)
        # 1/0

        ob_batch = init_ob_batch


        done = False
        iter = 0

        while not done:
            # print("Iter: {}".format(iter))
            # obs.append(ob)
            # infos.append(info)

            policies = torch.zeros((self.n_agents, self.batch_size, 1))
            actions = torch.zeros((self.n_agents, self.batch_size, 1), dtype=int)

            # print(ob[0][0])
            # print(ob)

            for i in range(self.NUM_AGENTS):
                ob[i] = torch.FloatTensor(ob[i])


            if self.history_len > 1:
                new_ob = ob_batch.clone()

                for i in range(self.NUM_AGENTS):

                    new_ob[i, :,
                    :self.dims_no_history[0] * (self.history_len - 1)] = \
                        ob_batch[i, :,
                        self.dims_no_history[0]:self.dims_no_history[
                                                    0] * self.history_len]

                    new_ob[i, :,
                    self.dims_no_history[0] * (self.history_len - 1):] = ob[
                        i]

                ob_batch = new_ob

            for i in range(self.NUM_AGENTS):

                if self.history_len > 1:

                    policy = th[i](ob_batch[i])  # Should be 4-D
                    # print(policy.shape)

                    state_value = vals[i](ob_batch[i])
                else:
                    policy = th[i](ob[i])  # Should be 4-D
                    # print(policy.shape)



                    state_value = vals[i](ob[i])
                    # print(state_value)

                    # print(policy)

                # action = torch.distributions.categorical.Categorical(
                #     probs=policy.detach()).sample()
                action = torch.distributions.binomial.Binomial(
                    probs=policy.detach()).sample()

                # print(policy)
                # print(policies)
                # print(action)
                # 1/0
                # print(policy.shape)
                # print(action.shape)

                # policy = policy.squeeze(1)
                # action = action.squeeze(-1)

                # action_prob = policy[
                #     torch.arange(0, self.batch_size, 1), action.long()]

                # print(action_prob.shape)

                # action_prob = policy[:, action.unsqueeze(dim=1)]
                # action_prob = torch.gather(policy, 0, action.unsqueeze(dim=1))

                # print(action_prob)

                obs_history[iter][i] = ob[i]

                val_history[iter][i] = state_value

                # print(policies.shape)
                # print(policy.shape)

                # policies[i] = action_prob
                policies[i] = policy

                # policy_history[iter][i] = action_prob
                actions[i] = action

            policy_history[iter] = policies

            # TODO update actions here with each individual action

            # Sample from categorical distribution here (th assumed to be nn that outputs 4-d softmax prob)

            act_history[iter] = actions  # actions as in action of every agent

            # print(actions)
            # print(rewards)

            # print("---")
            # for i in range(self.NUM_AGENTS):
            #     print(ob[i].reshape(self.batch_size, 4, 3, 3))
            # print(actions)
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

        next_val_history = self.get_next_val_history(val_history,
                                                     ending_state_values=state_value)  # iter doesn't even matter here as long as > 0

        return obs_history.unsqueeze(-1), act_history.unsqueeze(-1), rewards.unsqueeze(-1), \
               policy_history.unsqueeze(-1), val_history, next_val_history


    def get_next_val_history(self, val_history, ending_state_values):
        # The notation and naming here is a bit questionable. Vals is the actual parameterized value function
        # Val_history or state_vals as in some of the other functions are the state values for the given states in
        # some rollout/act_history

        next_val_history = torch.zeros(
            (self.max_steps, self.n_agents, self.batch_size, 1))
        next_val_history[:self.max_steps - 1, :, :, :] = \
            val_history[1:self.max_steps, :, :, :]
        next_val_history[-1, :, :, :] = ending_state_values

        return next_val_history



    def get_loss_helper(self, act_history, rewards, policy_history):
        num_iters = self.max_steps

        discounts = torch.cumprod(self.gamma * torch.ones((num_iters)),
                                  dim=0) / self.gamma


        # Discounted rewards, not sum of rewards which G_t is
        gamma_t_r_ts = rewards * discounts.reshape(-1, 1, 1,
                                                   1)  # implicit broadcasting done by numpy



        G_ts = reverse_cumsum(gamma_t_r_ts, dim=0)

        p_act_given_state = act_history.float() * policy_history + (
                1 - act_history.float()) * (1 - policy_history)


        log_p_act = torch.log(p_act_given_state)
        return G_ts, gamma_t_r_ts, log_p_act, discounts



    def get_dice_loss(self, act_history, rewards, policy_history, val_history, next_val_history, use_nl_loss=False):


        raise Exception("refactor this code and use the same/consistent one with IPD, especially considering modifications I made to the repeat_train code, and move it to a separate function so you don't have duplicate code")

        G_ts, gamma_t_r_ts, log_p_act_or_p_act_ratio, discounts = self.get_loss_helper(act_history,
            rewards, policy_history)

        discounts = discounts.view(-1, 1, 1, 1)

        R_ts = G_ts / discounts

        # print(rewards.shape)
        # print(val_history.shape)
        # print(next_val_history.shape)

        advantages = rewards + self.gamma * next_val_history - val_history


        sum_over_agents_log_p_act_or_p_act_ratio = log_p_act_or_p_act_ratio.sum(dim=1)
        # print(sum_over_agents_log_p_act_or_p_act_ratio.shape)

        deps_up_to_t = (torch.cumsum(sum_over_agents_log_p_act_or_p_act_ratio, dim=0)).reshape(-1, 1, self.batch_size, 1)

        deps_less_than_t = deps_up_to_t - sum_over_agents_log_p_act_or_p_act_ratio.reshape(-1, 1, self.batch_size, 1) # take out the dependency in the given time step

        # Look at Loaded DiCE paper to see where this formulation comes from
        loaded_dice_rewards = ((magic_box(deps_up_to_t) - magic_box(deps_less_than_t)) * discounts * advantages).sum(dim=0).mean(dim=1)


        dice_loss = -loaded_dice_rewards

        final_state_vals = next_val_history[-1]

        # print(R_ts.shape)

        values_loss = ((R_ts + (self.gamma * discounts.flip(dims=[0])) * final_state_vals.reshape(1, *final_state_vals.shape) - val_history) ** 2).sum(dim=0).mean(dim=1)

        if use_nl_loss:
            # No LOLA/opponent shaping or whatever, just naive learning
            # But this is not right because we aren't using the advantage estimation scheme.
            regular_nl_loss = -(log_p_act_or_p_act_ratio * advantages).sum(dim=0).mean(dim=1)
            return regular_nl_loss, G_ts, values_loss

        return dice_loss, G_ts, values_loss


    def print_policy_and_value_info(self, th, vals):
        state_batch = torch.tensor([[1,0,0,0,0],
                                    [0, 1, 0, 0, 0],
                                    [0, 0, 1, 0, 0],
                                    [0, 0, 0, 1, 0],
                                    [0, 0, 0, 0, 1],
                                    ], dtype=torch.float32)
        for i in range(len(th)):
            if isinstance(th[i], torch.Tensor):
                policy = torch.sigmoid(th[i])

            else:
                policy = th[i](state_batch)

            self.print_policy_info(policy, i)

        for i in range(len(vals)):
            print("Values {}".format(i + 1))
            if isinstance(vals[i], torch.Tensor):
                values = vals[i]
            else:
                values = vals[i](state_batch)
            print(values)
