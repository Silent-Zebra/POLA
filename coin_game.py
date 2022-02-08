"""
Coin Game environment. Adapted from https://github.com/alshedivat/lola/blob/master/lola_dice/envs/coin_game.py
"""
import gym
import numpy as np
import torch
from LOLA_nplayers_rollouts import magic_box, reverse_cumsum, Game

# from gym.spaces import Discrete, Tuple
# from gym.spaces import prng


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




class CoinGameGPU(Game):
    # No history len supported here (just RNN/full seq obs)
    """
    Vectorized Coin Game environment.
    Note: slightly deviates from the Gym API.
    """
    NUM_AGENTS = 2
    NUM_ACTIONS = 4
    MOVES = torch.stack([
        torch.LongTensor([0, 1]),
        torch.LongTensor([0, -1]),
        torch.LongTensor([1, 0]),
        torch.LongTensor([-1, 0]),
    ], dim=0).to(device)

    def __init__(self, max_steps, batch_size, grid_size=3, gamma=0.96):
        self.max_steps = max_steps
        self.grid_size = grid_size
        self.batch_size = batch_size
        # The 4 channels stand for 2 players and 2 coin positions
        self.ob_space_shape = [4, grid_size, grid_size]
        self.NUM_STATES = np.prod(self.ob_space_shape)
        self.available_actions = 4
        self.step_count = None
        self.full_seq_obs = True
        self.dims_with_history = [4 * grid_size * grid_size] * self.NUM_AGENTS
        self.n_agents = self.NUM_AGENTS
        self.dims_no_history = [4 * grid_size * grid_size] * self.NUM_AGENTS
        self.gamma = gamma



    def reset(self):
        self.step_count = 0

        red_pos_flat = torch.randint(self.grid_size * self.grid_size,
                                     size=(self.batch_size,)).to(device)
        self.red_pos = torch.stack(
            ( torch.div(red_pos_flat, self.grid_size, rounding_mode='floor')  , red_pos_flat % self.grid_size),
            dim=-1)

        blue_pos_flat = torch.randint(self.grid_size * self.grid_size,
                                      size=(self.batch_size,)).to(device)
        #         blue_pos_flat[blue_pos_flat >= red_pos_flat] += 1
        self.blue_pos = torch.stack(
            ( torch.div(blue_pos_flat, self.grid_size, rounding_mode='floor') , blue_pos_flat % self.grid_size),
            dim=-1)

        red_coin_pos_flat = torch.randint(self.grid_size * self.grid_size,
                                          size=(self.batch_size,)).to(device)
        blue_coin_pos_flat = torch.randint(self.grid_size * self.grid_size,
                                           size=(self.batch_size,)).to(device)
        #         minpos = torch.min(red_pos_flat, blue_pos_flat)
        #         maxpos = torch.max(red_pos_flat, blue_pos_flat)
        #         coin_pos_flat[coin_pos_flat >= minpos] += 1
        #         coin_pos_flat[coin_pos_flat >= maxpos] += 1

        self.red_coin_pos = torch.stack((torch.div(red_coin_pos_flat, self.grid_size, rounding_mode='floor'),
                                         red_coin_pos_flat % self.grid_size),
                                        dim=-1)
        self.blue_coin_pos = torch.stack((torch.div(blue_coin_pos_flat, self.grid_size, rounding_mode='floor'),
                                          blue_coin_pos_flat % self.grid_size),
                                         dim=-1)

        state = self._generate_state()
        state = state.reshape(self.batch_size, -1)
        observations = torch.stack([state, state])
        return observations

    def _generate_coins(self):
        mask_red = torch.logical_or(
            self._same_pos(self.red_coin_pos, self.blue_pos),
            self._same_pos(self.red_coin_pos, self.red_pos))
        red_coin_pos_flat = torch.randint(self.grid_size * self.grid_size,
                                          size=(self.batch_size,)).to(device)[
            mask_red]
        self.red_coin_pos[mask_red] = torch.stack((torch.div(red_coin_pos_flat, self.grid_size, rounding_mode='floor'),
                                                  red_coin_pos_flat % self.grid_size),
                                                  dim=-1)

        mask_blue = torch.logical_or(
            self._same_pos(self.blue_coin_pos, self.blue_pos),
            self._same_pos(self.blue_coin_pos, self.red_pos))
        blue_coin_pos_flat = torch.randint(self.grid_size * self.grid_size,
                                           size=(self.batch_size,)).to(device)[
            mask_blue]
        self.blue_coin_pos[mask_blue] = torch.stack((torch.div(blue_coin_pos_flat, self.grid_size, rounding_mode='floor'),
                                                    blue_coin_pos_flat % self.grid_size),
                                                    dim=-1)

    def _same_pos(self, x, y):
        return torch.all(x == y, dim=-1)

    def _generate_state(self):
        red_pos_flat = self.red_pos[:, 0] * self.grid_size + self.red_pos[:, 1]
        blue_pos_flat = self.blue_pos[:, 0] * self.grid_size + self.blue_pos[:,
                                                               1]

        red_coin_pos_flat = self.red_coin_pos[:,
                            0] * self.grid_size + self.red_coin_pos[:, 1]
        blue_coin_pos_flat = self.blue_coin_pos[:,
                             0] * self.grid_size + self.blue_coin_pos[:, 1]

        state = torch.zeros(
            (self.batch_size, 4, self.grid_size * self.grid_size)).to(device)

        #         state.scatter_(1, coin_pos_flatter[:,None], 1)
        #         state = state.view((self.batch_size, 4, self.grid_size*self.grid_size))

        state[:, 0].scatter_(1, red_pos_flat[:, None], 1)
        state[:, 1].scatter_(1, blue_pos_flat[:, None], 1)
        state[:, 2].scatter_(1, red_coin_pos_flat[:, None], 1)
        state[:, 3].scatter_(1, blue_coin_pos_flat[:, None], 1)

        return state.view(self.batch_size, 4, self.grid_size, self.grid_size).view(self.batch_size, 1, -1)

    def step(self, actions):
        ac0, ac1 = actions

        self.step_count += 1

        self.red_pos = (self.red_pos + self.MOVES[ac0]) % self.grid_size
        self.blue_pos = (self.blue_pos + self.MOVES[ac1]) % self.grid_size

        # Compute rewards
        red_reward = torch.zeros(self.batch_size).to(device)
        red_red_matches = self._same_pos(self.red_pos, self.red_coin_pos)
        red_reward[red_red_matches] += 1
        red_blue_matches = self._same_pos(self.red_pos, self.blue_coin_pos)
        red_reward[red_blue_matches] += 1

        blue_reward = torch.zeros(self.batch_size).to(device)
        blue_red_matches = self._same_pos(self.blue_pos, self.red_coin_pos)
        blue_reward[blue_red_matches] += 1
        blue_blue_matches = self._same_pos(self.blue_pos, self.blue_coin_pos)
        blue_reward[blue_blue_matches] += 1

        red_reward[blue_red_matches] -= 2
        blue_reward[red_blue_matches] -= 2

        self._generate_coins()
        reward = [red_reward.float(), blue_reward.float()]
        state = self._generate_state()
        observations = torch.stack((state, state))
        if self.step_count >= self.max_steps:
            done = torch.ones(self.batch_size).to(device)
        else:
            done = torch.zeros(self.batch_size).to(device)

        return observations, reward, done, (
        red_red_matches.sum(), red_blue_matches.sum(), blue_red_matches.sum(),
        blue_blue_matches.sum())

    def get_next_val_history(self, val_history, ending_state_values):
        # The notation and naming here is a bit questionable. Vals is the actual parameterized value function
        # Val_history or state_vals as in some of the other functions are the state values for the given states in
        # some rollout/trajectory

        next_val_history = torch.zeros(
            (self.max_steps, self.n_agents, self.batch_size, 1), device=device)
        next_val_history[:self.max_steps - 1, :, :, :] = \
            val_history[1:self.max_steps, :, :, :]
        next_val_history[-1, :, :, :] = ending_state_values

        return next_val_history

    def print_info_on_sample_obs(self, sample_obs, th, vals):

        if self.full_seq_obs:
            sample_obs = sample_obs.reshape(-1, 1, 36).to(device)

        for i in range(self.NUM_AGENTS):
            print("Agent {}:".format(i + 1))
            # with torch.backends.cudnn.flags(enabled=False):
            policy = th[i](sample_obs)
            value = vals[i](sample_obs)
            print(policy)
            print(value)

    def print_policy_and_value_info(self, th, vals):
        # Policy test
        if self.full_seq_obs:

            print("Simple One Step Example")
            sample_obs = torch.FloatTensor([[[0, 0, 0],
                                             [0, 1, 0],
                                             [0, 0, 0]],  # agent 1
                                            [[0, 0, 0],
                                             [1, 0, 0],
                                             [0, 0, 0]],  # agent 2
                                            [[0, 0, 0],
                                             [0, 0, 1],
                                             [0, 0, 0]],
                                            # red coin - so we should ideally want agent 1 to move right and agent 2 to not move left
                                            [[0, 0, 0],
                                             [0, 0, 0],
                                             [0, 0, 0]]]).reshape(1, 36)

            self.print_info_on_sample_obs(sample_obs, th, vals)

            # This one meant to test the idea of p2 defects by taking p1 coin - will p1 retaliate?
            print("P2 Defects")
            sample_obs_1 = torch.FloatTensor([[[1, 0, 0],
                                               [0, 0, 0],
                                               [0, 0, 0]],  # agent 1
                                              [[0, 0, 0],
                                               [0, 0, 0],
                                               [0, 1, 0]],  # agent 2
                                              [[0, 0, 0],
                                               [0, 0, 0],
                                               [1, 0, 0]],
                                              # red coin
                                              [[0, 0, 0],
                                               [0, 0, 0],
                                               [0, 0, 0]]]).reshape(1, 36)
            sample_obs_2 = torch.FloatTensor([[[0, 0, 0],
                                               [1, 0, 0],
                                               [0, 0, 0]],  # agent 1
                                              [[0, 0, 0],
                                               [0, 0, 0],
                                               [1, 0, 0]],  # agent 2
                                              [[0, 0, 0],
                                               [0, 0, 0],
                                               [0, 0, 0]],
                                              # red coin
                                              [[0, 0, 0],
                                               [0, 1, 0],
                                               [0, 0, 0]]]).reshape(1, 36)
            sample_obs = torch.stack((sample_obs_1, sample_obs_2), dim=1)

            self.print_info_on_sample_obs(sample_obs, th, vals)

            # This one meant similar to above except p2 cooperates by not taking coin.
            # Then p1 collects p1 coin (red). Will it also collect the other agent coin?
            print("P2 Cooperates")
            sample_obs_1 = torch.FloatTensor([[[1, 0, 0],
                                               [0, 0, 0],
                                               [0, 0, 0]],  # agent 1
                                              [[0, 0, 0],
                                               [0, 0, 0],
                                               [0, 1, 0]],  # agent 2
                                              [[0, 0, 0],
                                               [0, 0, 0],
                                               [1, 0, 0]],
                                              # red coin
                                              [[0, 0, 0],
                                               [0, 0, 0],
                                               [0, 0, 0]]]).reshape(1, 36)
            sample_obs_2 = torch.FloatTensor([[[0, 0, 0],
                                               [1, 0, 0],
                                               [0, 0, 0]],  # agent 1
                                              [[0, 0, 0],
                                               [0, 1, 0],
                                               [0, 0, 0]],  # agent 2
                                              [[0, 0, 0],
                                               [0, 0, 0],
                                               [1, 0, 0]],
                                              # red coin
                                              [[0, 0, 0],
                                               [0, 0, 0],
                                               [0, 0, 0]]]).reshape(1, 36)
            sample_obs_3 = torch.FloatTensor([[[0, 0, 0],
                                               [0, 0, 0],
                                               [1, 0, 0]],  # agent 1
                                              [[0, 1, 0],
                                               [0, 0, 0],
                                               [0, 0, 0]],  # agent 2
                                              [[0, 0, 0],
                                               [0, 0, 0],
                                               [0, 0, 0]],
                                              # red coin
                                              [[0, 0, 0],
                                               [0, 0, 0],
                                               [0, 1, 0]]]).reshape(1, 36)
            # Want to see prob of going right going down.
            sample_obs = torch.stack((sample_obs_1, sample_obs_2, sample_obs_3),
                                     dim=1)

            self.print_info_on_sample_obs(sample_obs, th, vals)

            print("P1 Defects")
            # This one meant to test the idea of p1 defects by taking p2 coin - will p2 retaliate?
            sample_obs_1 = torch.FloatTensor([[[1, 0, 0],
                                               [0, 0, 0],
                                               [0, 0, 0]],  # agent 1
                                              [[0, 0, 0],
                                               [0, 0, 0],
                                               [0, 1, 0]],  # agent 2
                                              [[0, 0, 0],
                                               [0, 0, 0],
                                               [0, 0, 0]],
                                              # red coin
                                              [[0, 0, 0],
                                               [1, 0, 0],
                                               [0, 0, 0]]]).reshape(1, 36)
            sample_obs_2 = torch.FloatTensor([[[0, 0, 0],
                                               [1, 0, 0],
                                               [0, 0, 0]],  # agent 1
                                              [[0, 0, 0],
                                               [0, 0, 0],
                                               [1, 0, 0]],  # agent 2
                                              [[0, 0, 0],
                                               [0, 0, 0],
                                               [0, 1, 0]],
                                              # red coin
                                              [[0, 0, 0],
                                               [0, 0, 0],
                                               [0, 0, 0]]]).reshape(1, 36)
            sample_obs = torch.stack((sample_obs_1, sample_obs_2), dim=1)

            self.print_info_on_sample_obs(sample_obs, th, vals)

            print("P1 Cooperates")
            # This one meant similar to above except p1 cooperates by not taking coin.
            # Then p2 collects p2 coin (blue). Will it also collect the other agent coin?
            sample_obs_1 = torch.FloatTensor([[[1, 0, 0],
                                               [0, 0, 0],
                                               [0, 0, 0]],  # agent 1
                                              [[0, 0, 0],
                                               [0, 0, 0],
                                               [0, 1, 0]],  # agent 2
                                              [[0, 0, 0],
                                               [0, 0, 0],
                                               [0, 0, 0]],
                                              # red coin
                                              [[0, 0, 0],
                                               [1, 0, 0],
                                               [0, 0, 0]]]).reshape(1, 36)
            sample_obs_2 = torch.FloatTensor([[[0, 1, 0],
                                               [0, 0, 0],
                                               [0, 0, 0]],  # agent 1
                                              [[0, 0, 0],
                                               [0, 0, 0],
                                               [1, 0, 0]],  # agent 2
                                              [[0, 0, 0],
                                               [0, 0, 0],
                                               [0, 0, 0]],
                                              # red coin
                                              [[0, 0, 0],
                                               [1, 0, 0],
                                               [0, 0, 0]]]).reshape(1, 36)
            sample_obs_3 = torch.FloatTensor([[[0, 0, 1],
                                               [0, 0, 0],
                                               [0, 0, 0]],  # agent 1
                                              [[0, 0, 0],
                                               [1, 0, 0],
                                               [0, 0, 0]],  # agent 2
                                              [[0, 0, 0],
                                               [0, 1, 0],
                                               [0, 0, 0]],
                                              # red coin
                                              [[0, 0, 0],
                                               [0, 0, 0],
                                               [0, 0, 0]]]).reshape(1, 36)
            sample_obs = torch.stack(
                (sample_obs_1, sample_obs_2, sample_obs_3), dim=1)

            self.print_info_on_sample_obs(sample_obs, th, vals)

    def rollout(self, th, vals, gamma=0.96):
        # Assumes same dim obs for all players
        obs_history = torch.zeros((self.max_steps, self.n_agents,
                                   self.batch_size, self.dims_no_history[0]),
                                  dtype=torch.int, device=device)
        act_history = torch.zeros(
            (self.max_steps, self.n_agents, self.batch_size), dtype=torch.int, device=device)
        rewards = torch.zeros(
            (self.max_steps, self.n_agents, self.batch_size), device=device)
        policy_history = torch.zeros(
            (self.max_steps, self.n_agents, self.batch_size), device=device)
        val_history = torch.zeros(
            (self.max_steps, self.n_agents, self.batch_size, 1), device=device)

        ob = self.reset()

        if self.full_seq_obs:
            # init_ob_batch = torch.zeros((self.n_agents, self.batch_size, 1, self.dims_no_history[0]))
            init_ob_batch = None

        else:
            init_ob_batch = torch.zeros(
                (self.n_agents, self.batch_size, self.dims_with_history[0]), device=device)
            for i in range(len(ob)):
                # fill all with starting state
                start_counter = 0
                end_counter = self.dims_no_history[0]
                for j in range(self.history_len):
                    init_ob_batch[i, :,
                    start_counter:end_counter] = torch.FloatTensor(ob[i])
                    start_counter = end_counter
                    end_counter += self.dims_no_history[0]

        ob_batch = init_ob_batch

        done = False
        # gamma_t = 1.
        iter = 0

        avg_same_colour_coins_picked_total = 0
        avg_diff_colour_coins_picked_total = 0
        avg_coins_picked_total = 0

        while not done:

            # print("Iter: {}".format(iter))
            # obs.append(ob)
            # infos.append(info)

            policies = torch.zeros((self.n_agents, self.batch_size), device=device)
            actions = torch.zeros((self.n_agents, self.batch_size), dtype=int, device=device)

            # print(ob[0][0])

            # for i in range(self.NUM_AGENTS):
            #     ob[i] = torch.FloatTensor(ob[i])

            if self.full_seq_obs:
                if ob_batch is None:
                    # Need the unsqueeze for the sequence of obs, and then the reshaping, borrow that code. Try to keep code as consistent as possible.
                    # Basically unsqueeze gives the history dimension
                    # and the final dimension is 36 (4 x 3 x 3)
                    # print(ob.shape)
                    ob_batch = ob.unsqueeze(2)
                else:
                    # print(ob_batch)
                    # print(torch.stack(ob).unsqueeze(2).shape)
                    # print(ob.shape)
                    # print(ob_batch.shape)
                    new_ob = torch.cat((ob_batch, ob),
                                       dim=2)
                    # print(new_ob.shape)
                    # 1/0
                    # probably need just ob instead of torch.stack(ob).unsqueeze
                    # print(new_ob.shape)
                    ob_batch = new_ob

            else:
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

                if self.full_seq_obs:

                    # ob_batch[i] = ob_batch[i].reshape(ob_batch[i].shape[0], -1, ob_batch[i].shape[-1])
                    with torch.backends.cudnn.flags(enabled=False):
                        policy = th[i](ob_batch[i])  # Should be 4-D

                    state_value = vals[i](ob_batch[i])
                    # print(state_value.shape)

                    state_value = state_value[:, -1]
                    # print(state_value.shape)

                    state_value = state_value.unsqueeze(-1)



                else:

                    if self.history_len > 1:
                        with torch.backends.cudnn.flags(enabled=False):
                            policy = th[i](ob_batch[i])  # Should be 4-D
                        # print(policy.shape)

                        state_value = vals[i](ob_batch[i])
                    else:
                        with torch.backends.cudnn.flags(enabled=False):
                            policy = th[i](ob[i])  # Should be 4-D
                        # print(policy.shape)

                        state_value = vals[i](ob[i])
                        # print(state_value)

                        # print(policy)

                # TODO make sure that this works with batches
                action = torch.distributions.categorical.Categorical(
                    probs=policy.detach()).sample()

                # print(action)

                # print(policy.shape)
                # print(action.shape)

                # policy = policy.squeeze(1)
                # action = action.squeeze(-1)

                action_prob = policy[
                    torch.arange(0, self.batch_size, 1), action]

                # print(action_prob.shape)

                # action_prob = policy[:, action.unsqueeze(dim=1)]
                # action_prob = torch.gather(policy, 0, action.unsqueeze(dim=1))

                # print(action_prob)

                # print(ob[i].shape)
                # print(obs_history[iter][i].shape)

                obs_history[iter][i] = ob[i].squeeze(1)

                val_history[iter][i] = state_value

                policies[i] = action_prob
                # policy_history[iter][i] = action_prob
                actions[i] = action

            policy_history[iter] = policies

            # TODO update actions here with each individual action

            # Sample from categorical distribution here (th assumed to be nn that outputs 4-d softmax prob)

            act_history[iter] = actions  # actions as in action of every agent

            # print("---")
            # for i in range(self.NUM_AGENTS):
            #     print(ob[i].reshape(self.batch_size, 4, 3, 3))
            # print(ob)

            # Testing Only
            # actions[-1] = torch.zeros_like(actions[-1], dtype=int)

            # print(actions)

            ob, rew, done, info = self.step(actions)

            done = done[0]

            rr_matches, rb_matches, br_matches, bb_matches = info

            avg_same_colour_coins_picked_total += (rr_matches + bb_matches) / self.batch_size
            avg_diff_colour_coins_picked_total += (rb_matches + br_matches) / self.batch_size
            avg_coins_picked_total += (rr_matches + bb_matches + rb_matches + br_matches) / self.batch_size

            # for i in range(self.NUM_AGENTS):
            #     print(ob[i].reshape(self.batch_size, 4, 3, 3))
            # print(rew)

            for i in range(self.NUM_AGENTS):
                rewards[iter][i] = rew[i]
                # rewards[iter][i] = torch.FloatTensor(rew[i])
            # acs.append(ac)
            # rews.append(rew)
            # rets.append([r * gamma_t for r in rew])
            # gamma_t *= gamma
            iter += 1



        next_val_history = self.get_next_val_history(val_history,
                                                     ending_state_values=state_value)  # iter doesn't even matter here as long as > 0

        return obs_history.unsqueeze(-1), act_history.unsqueeze(
            -1), rewards.unsqueeze(-1), \
               policy_history.unsqueeze(-1), val_history, next_val_history, \
               avg_same_colour_coins_picked_total, avg_diff_colour_coins_picked_total, avg_coins_picked_total
        # next_val_history

    # TODO check - copied from nplayers_rollouts contrib game formulation
    def get_loss_helper(self, trajectory, rewards, policy_history,
                        old_policy_history=None):
        num_iters = len(trajectory)

        discounts = torch.cumprod(self.gamma * torch.ones((num_iters), device=device),
                                  dim=0) / self.gamma

        # Discounted rewards, not sum of rewards which G_t is
        gamma_t_r_ts = rewards * discounts.reshape(-1, 1, 1, 1)  # implicit broadcasting done by numpy

        G_ts = reverse_cumsum(gamma_t_r_ts, dim=0)
        # G_ts gives you the inner sum of discounted rewards

        p_act_given_state = trajectory.float() * policy_history + (
                1 - trajectory.float()) * (1 - policy_history)

        if old_policy_history is None:
            # recall 1 is coop, so when coop action 1 taken, we look at policy which is prob coop
            # and when defect 0 is taken, we take 1-policy = prob of defect
            log_p_act = torch.log(p_act_given_state)

            return G_ts, gamma_t_r_ts, log_p_act, discounts
        else:
            p_act_given_state_old = trajectory.float() * old_policy_history + (
                    1 - trajectory.float()) * (1 - old_policy_history)

            p_act_ratio = p_act_given_state / p_act_given_state_old.detach()

            return G_ts, gamma_t_r_ts, p_act_ratio, discounts

    def get_dice_loss(self, trajectory, rewards, policy_history, val_history,
                      next_val_history,
                      old_policy_history=None, kl_div_target_policy=None,
                      use_nl_loss=False, use_clipping=False, use_penalty=False,
                      beta=None):

        if old_policy_history is not None:
            old_policy_history = old_policy_history.detach()

        G_ts, gamma_t_r_ts, log_p_act_or_p_act_ratio, discounts = self.get_loss_helper(
            trajectory, rewards, policy_history, old_policy_history)

        discounts = discounts.view(-1, 1, 1, 1)

        # R_t is like G_t except not discounted back to the start. It is the forward
        # looking return at that point in time
        R_ts = G_ts / discounts

        # Generalized Advantage Estimation (GAE) calc adapted from loaded dice repo
        # https://github.com/oxwhirl/loaded-dice/blob/master/loaded_dice_demo.ipynb
        advantages = torch.zeros_like(G_ts, device=device)
        lambd = 0  # 0.95 # 1 here is essentially what I was doing before with monte carlo
        # print(rewards)
        # print(next_val_history)
        # print(val_history)
        deltas = rewards + self.gamma * next_val_history.detach() - val_history.detach()
        gae = torch.zeros_like(deltas[0, :], device=device).float()
        for i in range(deltas.size(0) - 1, -1, -1):
            gae = gae * self.gamma * lambd + deltas[i, :]
            advantages[i, :] = gae

        # if inner_repeat_train_on_same_samples:
        #     # Then we should have a p_act_ratio here instead of a log_p_act
        #     if use_clipping:
        #
        #         # Two way clamp, not yet ppo style
        #         if two_way_clip:
        #             log_p_act_or_p_act_ratio = torch.clamp(log_p_act_or_p_act_ratio, min=1 - clip_epsilon, max=1 + clip_epsilon)
        #         else:
        #             # PPO style clipping
        #             pos_adv = (advantages > 0).float()
        #             log_p_act_or_p_act_ratio = pos_adv * torch.minimum(log_p_act_or_p_act_ratio,torch.zeros_like(log_p_act_or_p_act_ratio) + 1+clip_epsilon) + \
        #                                        (1-pos_adv) * torch.maximum(log_p_act_or_p_act_ratio,torch.zeros_like(log_p_act_or_p_act_ratio) + 1-clip_epsilon)
        #
        #     if use_penalty:
        #         # Calculate KL Divergence
        #         kl_divs = torch.zeros((self.n_agents))
        #
        #         if kl_div_target_policy is None:
        #             assert old_policy_history is not None
        #             kl_div_target_policy = old_policy_history
        #
        #         for i in range(self.n_agents):
        #
        #             policy_dist_i = self.build_policy_dist(policy_history, i)
        #             kl_target_dist_i = self.build_policy_dist(kl_div_target_policy, i)
        #
        #             kl_div = torch.nn.functional.kl_div(input=torch.log(policy_dist_i),
        #                                             target=kl_target_dist_i.detach(),
        #                                             reduction='batchmean',
        #                                             log_target=False)
        #             # print(kl_div)
        #             kl_divs[i] = kl_div
        #
        #         # print(kl_divs)

        sum_over_agents_log_p_act_or_p_act_ratio = log_p_act_or_p_act_ratio.sum(
            dim=1)

        # See 5.2 (page 7) of DiCE paper for below:
        # With batches, the mean is the mean across batches. The sum is over the steps in the rollout/trajectory

        deps_up_to_t = (torch.cumsum(sum_over_agents_log_p_act_or_p_act_ratio,
                                     dim=0)).reshape(-1, 1, self.batch_size, 1)

        deps_less_than_t = deps_up_to_t - sum_over_agents_log_p_act_or_p_act_ratio.reshape(
            -1, 1, self.batch_size,
            1)  # take out the dependency in the given time step

        # Look at Loaded DiCE paper to see where this formulation comes from
        # Right now since I am using GAE, the advantages already have the discounts in them, no need to multiply again
        loaded_dice_rewards = ((magic_box(deps_up_to_t) - magic_box(
            deps_less_than_t)) * advantages).sum(dim=0).mean(dim=1)

        dice_loss = -loaded_dice_rewards

        # if inner_repeat_train_on_same_samples and use_penalty:
        #     kl_divs = kl_divs.unsqueeze(-1)
        #
        #     assert beta is not None
        #
        #     # TODO make adaptive
        #     dice_loss += beta * kl_divs # we want to min the positive kl_div

        final_state_vals = next_val_history[-1].detach()
        # You DO need a detach on these. Because it's the target - it should be detached. It's a target value.
        values_loss = ((R_ts + (self.gamma * discounts.flip(
            dims=[0])) * final_state_vals.reshape(1,
                                                  *final_state_vals.shape) - val_history) ** 2).sum(
            dim=0).mean(dim=1)

        if use_nl_loss:
            # No LOLA/opponent shaping or whatever, just naive learning
            regular_nl_loss = -(log_p_act_or_p_act_ratio * advantages).sum(
                dim=0).mean(dim=1)
            # Well I mean obviously if you do this there is no shaping because you can't differentiate through the inner update step...
            return regular_nl_loss, G_ts, values_loss

        return dice_loss, G_ts, values_loss


class SymmetricCoinGame:

    def __init__(self, b, inner_ep_len, gamma_inner=0.96):
        self.env = CoinGameGPU(max_steps=inner_ep_len - 1, batch_size=b)
        self.inner_ep_len = inner_ep_len
        self.b = b

    def reset(self):
        self.env_states = self.env.reset()[0]
        self.rewards_inner = torch.Tensor(np.array([0.0] * self.b)).to(device)
        self.rewards_outer = torch.Tensor(np.array([0.0] * self.b)).to(device)
        self.dones_inner = torch.Tensor(np.array([0.0] * self.b)).to(device)
        self.dones_outer = torch.Tensor(np.array([0.0] * self.b)).to(device)
        return self._prep_state()

    def _prep_state(self):

        rewards_inner_tiled = torch.tile(self.rewards_inner[None, None].T,
                                         [1, 3, 3])[:, None]
        rewards_outer_tiled = torch.tile(self.rewards_outer[None, None].T,
                                         [1, 3, 3])[:, None]
        dones_inner_tiled = torch.tile(self.dones_inner[None, None].T,
                                       [1, 3, 3])[:, None]
        env_states_outer = torch.stack(
            [self.env_states[:, 1], self.env_states[:, 0],
             self.env_states[:, 3], self.env_states[:, 2]], dim=1)
        return [
            torch.cat(
                [self.env_states, rewards_inner_tiled, rewards_outer_tiled,
                 dones_inner_tiled], axis=1),
            torch.cat(
                [env_states_outer, rewards_outer_tiled, rewards_inner_tiled,
                 dones_inner_tiled], axis=1),
        ]

    def step(self, actions):
        if torch.any(self.dones_inner):
            info = None
            self.env_states = self.env.reset()[0]
            self.rewards_inner = torch.Tensor(np.array([0.0] * self.b)).to(
                device)
            self.rewards_outer = torch.Tensor(np.array([0.0] * self.b)).to(
                device)
            self.dones_inner = torch.Tensor(np.array([0.0] * self.b)).to(device)
        else:
            self.env_states, rewards, self.dones_inner, info = self.env.step(
                actions
            )
            self.env_states = self.env_states[0]
            self.rewards_inner, self.rewards_outer = rewards
        return self._prep_state(), [self.rewards_inner,
                                    self.rewards_outer], info


class CoinGameVec(gym.Env, Game):
    """
    Vectorized Coin Game environment.
    Note: slightly deviates from the Gym API.
    """
    NUM_AGENTS = 2
    COIN_POSITIONS = NUM_AGENTS
    NUM_ACTIONS = 4
    MOVES = [
        np.array([0,  1]), # right
        np.array([0, -1]), # left
        np.array([1,  0]), # down
        np.array([-1, 0]), # up
    ]

    # def __init__(self, max_steps, batch_size, grid_size=3, num_agents=2):
    def __init__(self, max_steps, batch_size, history_len, grid_size=3, gamma=0.96, full_seq_obs=True):
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
        self.full_seq_obs = full_seq_obs


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
        # Note the way the coin game is set up, only one coin is on the field at any point in time.
        # As soon as it is collected, a new one of the opposite colour is spawned
        # This thing below alternates to keep track of the colour to spawn
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

        same_colour_coins_picked_this_step = 0
        diff_colour_coins_picked_this_step = 0


        for i in range(self.batch_size):
            generate = False
            if self.red_coin[i]:
                if self._same_pos(self.red_pos[i], self.coin_pos[i]):
                    generate = True
                    reward_red[i] += 1
                    same_colour_coins_picked_this_step += 1
                if self._same_pos(self.blue_pos[i], self.coin_pos[i]):
                    generate = True
                    reward_red[i] += -2
                    reward_blue[i] += 1
                    diff_colour_coins_picked_this_step += 1
            else:
                if self._same_pos(self.red_pos[i], self.coin_pos[i]):
                    generate = True
                    reward_red[i] += 1
                    reward_blue[i] += -2
                    diff_colour_coins_picked_this_step += 1
                if self._same_pos(self.blue_pos[i], self.coin_pos[i]):
                    generate = True
                    reward_blue[i] += 1
                    same_colour_coins_picked_this_step += 1

            if generate:
                self._generate_coin(i)

        coins_picked_this_step = same_colour_coins_picked_this_step + diff_colour_coins_picked_this_step

        avg_same_colour_coins = same_colour_coins_picked_this_step / self.batch_size
        avg_diff_colour_coins = diff_colour_coins_picked_this_step / self.batch_size
        avg_coins_picked_this_step = coins_picked_this_step / self.batch_size

        reward = [reward_red, reward_blue]
        state = self._generate_state().reshape((self.batch_size, -1))
        observations = [state, state]
        done = (self.step_count == self.max_steps)
        info = [{'available_actions': aa} for aa in self.available_actions]

        return observations, reward, done, info, avg_same_colour_coins, avg_diff_colour_coins, avg_coins_picked_this_step



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


    def print_info_on_sample_obs(self, sample_obs, th, vals):

        if self.full_seq_obs:
            sample_obs = sample_obs.reshape(-1, 1, 36)


        for i in range(self.NUM_AGENTS):
            print("Agent {}:".format(i+1))
            policy = th[i](sample_obs)
            value = vals[i](sample_obs)
            print(policy)
            print(value)


    def print_policy_and_value_info(self, th, vals):
        # Policy test
        if self.full_seq_obs:

            print("Simple One Step Example")
            sample_obs = torch.FloatTensor([[[0, 0, 0],
                                             [0, 1, 0],
                                             [0, 0, 0]],  # agent 1
                                            [[0, 0, 0],
                                             [1, 0, 0],
                                             [0, 0, 0]],  # agent 2
                                            [[0, 0, 0],
                                             [0, 0, 1],
                                             [0, 0, 0]],
                                            # red coin - so we should ideally want agent 1 to move right and agent 2 to not move left
                                            [[0, 0, 0],
                                             [0, 0, 0],
                                             [0, 0, 0]]]).reshape(1, 36)

            self.print_info_on_sample_obs(sample_obs, th, vals)


            # This one meant to test the idea of p2 defects by taking p1 coin - will p1 retaliate?
            print("P2 Defects")
            sample_obs_1 = torch.FloatTensor([[[1, 0, 0],
                                             [0, 0, 0],
                                             [0, 0, 0]],  # agent 1
                                            [[0, 0, 0],
                                             [0, 0, 0],
                                             [0, 1, 0]],  # agent 2
                                            [[0, 0, 0],
                                             [0, 0, 0],
                                             [1, 0, 0]],
                                            # red coin
                                            [[0, 0, 0],
                                             [0, 0, 0],
                                             [0, 0, 0]]]).reshape(1, 36)
            sample_obs_2 = torch.FloatTensor([[[0, 0, 0],
                                               [1, 0, 0],
                                               [0, 0, 0]],  # agent 1
                                              [[0, 0, 0],
                                               [0, 0, 0],
                                               [1, 0, 0]],  # agent 2
                                              [[0, 0, 0],
                                               [0, 0, 0],
                                               [0, 0, 0]],
                                              # red coin
                                              [[0, 0, 0],
                                               [0, 1, 0],
                                               [0, 0, 0]]]).reshape(1, 36)
            sample_obs = torch.stack((sample_obs_1, sample_obs_2), dim=1)

            self.print_info_on_sample_obs(sample_obs, th, vals)

            # This one meant similar to above except p2 cooperates by not taking coin.
            # Then p1 collects p1 coin (red). Will it also collect the other agent coin?
            print("P2 Cooperates")
            sample_obs_1 = torch.FloatTensor([[[1, 0, 0],
                                               [0, 0, 0],
                                               [0, 0, 0]],  # agent 1
                                              [[0, 0, 0],
                                               [0, 0, 0],
                                               [0, 1, 0]],  # agent 2
                                              [[0, 0, 0],
                                               [0, 0, 0],
                                               [1, 0, 0]],
                                              # red coin
                                              [[0, 0, 0],
                                               [0, 0, 0],
                                               [0, 0, 0]]]).reshape(1, 36)
            sample_obs_2 = torch.FloatTensor([[[0, 0, 0],
                                               [1, 0, 0],
                                               [0, 0, 0]],  # agent 1
                                              [[0, 0, 0],
                                               [0, 1, 0],
                                               [0, 0, 0]],  # agent 2
                                              [[0, 0, 0],
                                               [0, 0, 0],
                                               [1, 0, 0]],
                                              # red coin
                                              [[0, 0, 0],
                                               [0, 0, 0],
                                               [0, 0, 0]]]).reshape(1, 36)
            sample_obs_3 = torch.FloatTensor([[[0, 0, 0],
                                               [0, 0, 0],
                                               [1, 0, 0]],  # agent 1
                                              [[0, 1, 0],
                                               [0, 0, 0],
                                               [0, 0, 0]],  # agent 2
                                              [[0, 0, 0],
                                               [0, 0, 0],
                                               [0, 0, 0]],
                                              # red coin
                                              [[0, 0, 0],
                                               [0, 0, 0],
                                               [0, 1, 0]]]).reshape(1, 36)
            # Want to see prob of going right going down.
            sample_obs = torch.stack((sample_obs_1, sample_obs_2, sample_obs_3), dim=1)

            self.print_info_on_sample_obs(sample_obs, th, vals)

            print("P1 Defects")
            # This one meant to test the idea of p1 defects by taking p2 coin - will p2 retaliate?
            sample_obs_1 = torch.FloatTensor([[[1, 0, 0],
                                               [0, 0, 0],
                                               [0, 0, 0]],  # agent 1
                                              [[0, 0, 0],
                                               [0, 0, 0],
                                               [0, 1, 0]],  # agent 2
                                              [[0, 0, 0],
                                               [0, 0, 0],
                                               [0, 0, 0]],
                                              # red coin
                                              [[0, 0, 0],
                                               [1, 0, 0],
                                               [0, 0, 0]]]).reshape(1, 36)
            sample_obs_2 = torch.FloatTensor([[[0, 0, 0],
                                               [1, 0, 0],
                                               [0, 0, 0]],  # agent 1
                                              [[0, 0, 0],
                                               [0, 0, 0],
                                               [1, 0, 0]],  # agent 2
                                              [[0, 0, 0],
                                               [0, 0, 0],
                                               [0, 1, 0]],
                                              # red coin
                                              [[0, 0, 0],
                                               [0, 0, 0],
                                               [0, 0, 0]]]).reshape(1, 36)
            sample_obs = torch.stack((sample_obs_1, sample_obs_2), dim=1)

            self.print_info_on_sample_obs(sample_obs, th, vals)

            print("P1 Cooperates")
            # This one meant similar to above except p1 cooperates by not taking coin.
            # Then p2 collects p2 coin (blue). Will it also collect the other agent coin?
            sample_obs_1 = torch.FloatTensor([[[1, 0, 0],
                                               [0, 0, 0],
                                               [0, 0, 0]],  # agent 1
                                              [[0, 0, 0],
                                               [0, 0, 0],
                                               [0, 1, 0]],  # agent 2
                                              [[0, 0, 0],
                                               [0, 0, 0],
                                               [0, 0, 0]],
                                              # red coin
                                              [[0, 0, 0],
                                               [1, 0, 0],
                                               [0, 0, 0]]]).reshape(1, 36)
            sample_obs_2 = torch.FloatTensor([[[0, 1, 0],
                                               [0, 0, 0],
                                               [0, 0, 0]],  # agent 1
                                              [[0, 0, 0],
                                               [0, 0, 0],
                                               [1, 0, 0]],  # agent 2
                                              [[0, 0, 0],
                                               [0, 0, 0],
                                               [0, 0, 0]],
                                              # red coin
                                              [[0, 0, 0],
                                               [1, 0, 0],
                                               [0, 0, 0]]]).reshape(1, 36)
            sample_obs_3 = torch.FloatTensor([[[0, 0, 1],
                                               [0, 0, 0],
                                               [0, 0, 0]],  # agent 1
                                              [[0, 0, 0],
                                               [1, 0, 0],
                                               [0, 0, 0]],  # agent 2
                                              [[0, 0, 0],
                                               [0, 1, 0],
                                               [0, 0, 0]],
                                              # red coin
                                              [[0, 0, 0],
                                               [0, 0, 0],
                                               [0, 0, 0]]]).reshape(1, 36)
            sample_obs = torch.stack(
                (sample_obs_1, sample_obs_2, sample_obs_3), dim=1)

            self.print_info_on_sample_obs(sample_obs, th, vals)



    def rollout(self, th, vals, gamma=0.96):
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


        if self.full_seq_obs:
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


        avg_same_colour_coins_picked_total = 0
        avg_diff_colour_coins_picked_total = 0
        avg_coins_picked_total = 0

        while not done:
            # print("Iter: {}".format(iter))
            # obs.append(ob)
            # infos.append(info)

            policies = torch.zeros((self.n_agents, self.batch_size))
            actions = torch.zeros((self.n_agents, self.batch_size),dtype=int)


            # print(ob[0][0])


            for i in range(self.NUM_AGENTS):
                ob[i] = torch.FloatTensor(ob[i])

            if self.full_seq_obs:
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

                if self.full_seq_obs:

                    # print(ob_batch[i].shape)
                    # print(th[i])

                    policy = th[i](ob_batch[i])  # Should be 4-D

                    state_value = vals[i](ob_batch[i])
                    # print(state_value.shape)

                    state_value = state_value[:, -1]
                    # print(state_value.shape)

                    state_value = state_value.unsqueeze(-1)



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

                # print(policy.shape)
                # print(action.shape)

                # policy = policy.squeeze(1)
                # action = action.squeeze(-1)

                action_prob = policy[torch.arange(0,self.batch_size,1), action]

                # print(action_prob.shape)


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

            ob, rew, done, info, avg_same_colour_coins, avg_diff_colour_coins, avg_coins_picked_this_step = self.step(actions.numpy())


            avg_same_colour_coins_picked_total += avg_same_colour_coins
            avg_diff_colour_coins_picked_total += avg_diff_colour_coins
            avg_coins_picked_total += avg_coins_picked_this_step

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


        next_val_history = self.get_next_val_history(val_history, ending_state_values=state_value) # iter doesn't even matter here as long as > 0

        return obs_history.unsqueeze(-1), act_history.unsqueeze(-1), rewards.unsqueeze(-1), \
               policy_history.unsqueeze(-1), val_history, next_val_history, \
               avg_same_colour_coins_picked_total, avg_diff_colour_coins_picked_total, avg_coins_picked_total
               # next_val_history

    # TODO check - copied from nplayers_rollouts contrib game formulation
    def get_loss_helper(self, trajectory, rewards, policy_history,
                        old_policy_history=None):
        num_iters = len(trajectory)

        discounts = torch.cumprod(self.gamma * torch.ones((num_iters)),
                                  dim=0) / self.gamma

        # Discounted rewards, not sum of rewards which G_t is
        gamma_t_r_ts = rewards * discounts.reshape(-1, 1, 1,
                                                   1)  # implicit broadcasting done by numpy

        G_ts = reverse_cumsum(gamma_t_r_ts, dim=0)
        # G_ts gives you the inner sum of discounted rewards

        p_act_given_state = trajectory.float() * policy_history + (
                1 - trajectory.float()) * (1 - policy_history)

        if old_policy_history is None:
            # recall 1 is coop, so when coop action 1 taken, we look at policy which is prob coop
            # and when defect 0 is taken, we take 1-policy = prob of defect
            log_p_act = torch.log(p_act_given_state)

            return G_ts, gamma_t_r_ts, log_p_act, discounts
        else:
            p_act_given_state_old = trajectory.float() * old_policy_history + (
                    1 - trajectory.float()) * (1 - old_policy_history)

            p_act_ratio = p_act_given_state / p_act_given_state_old.detach()

            return G_ts, gamma_t_r_ts, p_act_ratio, discounts


    def get_dice_loss(self, trajectory, rewards, policy_history, val_history, next_val_history,
                      old_policy_history=None, kl_div_target_policy=None, use_nl_loss=False, use_clipping=False, use_penalty=False, beta=None):

        if old_policy_history is not None:
            old_policy_history = old_policy_history.detach()

        G_ts, gamma_t_r_ts, log_p_act_or_p_act_ratio, discounts = self.get_loss_helper(
            trajectory, rewards, policy_history, old_policy_history)

        discounts = discounts.view(-1, 1, 1, 1)

        # R_t is like G_t except not discounted back to the start. It is the forward
        # looking return at that point in time
        R_ts = G_ts / discounts

        # Generalized Advantage Estimation (GAE) calc adapted from loaded dice repo
        # https://github.com/oxwhirl/loaded-dice/blob/master/loaded_dice_demo.ipynb
        advantages = torch.zeros_like(G_ts)
        lambd = 0 #0.95 # 1 here is essentially what I was doing before with monte carlo
        deltas = rewards + self.gamma * next_val_history.detach() - val_history.detach()
        gae = torch.zeros_like(deltas[0,:]).float()
        for i in range(deltas.size(0) - 1, -1, -1):
            gae = gae * self.gamma * lambd + deltas[i,:]
            advantages[i,:] = gae

        # if inner_repeat_train_on_same_samples:
        #     # Then we should have a p_act_ratio here instead of a log_p_act
        #     if use_clipping:
        #
        #         # Two way clamp, not yet ppo style
        #         if two_way_clip:
        #             log_p_act_or_p_act_ratio = torch.clamp(log_p_act_or_p_act_ratio, min=1 - clip_epsilon, max=1 + clip_epsilon)
        #         else:
        #             # PPO style clipping
        #             pos_adv = (advantages > 0).float()
        #             log_p_act_or_p_act_ratio = pos_adv * torch.minimum(log_p_act_or_p_act_ratio,torch.zeros_like(log_p_act_or_p_act_ratio) + 1+clip_epsilon) + \
        #                                        (1-pos_adv) * torch.maximum(log_p_act_or_p_act_ratio,torch.zeros_like(log_p_act_or_p_act_ratio) + 1-clip_epsilon)
        #
        #     if use_penalty:
        #         # Calculate KL Divergence
        #         kl_divs = torch.zeros((self.n_agents))
        #
        #         if kl_div_target_policy is None:
        #             assert old_policy_history is not None
        #             kl_div_target_policy = old_policy_history
        #
        #         for i in range(self.n_agents):
        #
        #             policy_dist_i = self.build_policy_dist(policy_history, i)
        #             kl_target_dist_i = self.build_policy_dist(kl_div_target_policy, i)
        #
        #             kl_div = torch.nn.functional.kl_div(input=torch.log(policy_dist_i),
        #                                             target=kl_target_dist_i.detach(),
        #                                             reduction='batchmean',
        #                                             log_target=False)
        #             # print(kl_div)
        #             kl_divs[i] = kl_div
        #
        #         # print(kl_divs)

        sum_over_agents_log_p_act_or_p_act_ratio = log_p_act_or_p_act_ratio.sum(dim=1)

        # See 5.2 (page 7) of DiCE paper for below:
        # With batches, the mean is the mean across batches. The sum is over the steps in the rollout/trajectory

        deps_up_to_t = (torch.cumsum(sum_over_agents_log_p_act_or_p_act_ratio, dim=0)).reshape(-1, 1, self.batch_size, 1)

        deps_less_than_t = deps_up_to_t - sum_over_agents_log_p_act_or_p_act_ratio.reshape(-1, 1, self.batch_size, 1) # take out the dependency in the given time step

        # Look at Loaded DiCE paper to see where this formulation comes from
        # Right now since I am using GAE, the advantages already have the discounts in them, no need to multiply again
        loaded_dice_rewards = ((magic_box(deps_up_to_t) - magic_box(deps_less_than_t)) * advantages).sum(dim=0).mean(dim=1)

        dice_loss = -loaded_dice_rewards

        # if inner_repeat_train_on_same_samples and use_penalty:
        #     kl_divs = kl_divs.unsqueeze(-1)
        #
        #     assert beta is not None
        #
        #     # TODO make adaptive
        #     dice_loss += beta * kl_divs # we want to min the positive kl_div

        final_state_vals = next_val_history[-1].detach()
        # You DO need a detach on these. Because it's the target - it should be detached. It's a target value.
        values_loss = ((R_ts + (self.gamma * discounts.flip(dims=[0])) * final_state_vals.reshape(1, *final_state_vals.shape) - val_history) ** 2).sum(dim=0).mean(dim=1)

        if use_nl_loss:
            # No LOLA/opponent shaping or whatever, just naive learning
            regular_nl_loss = -(log_p_act_or_p_act_ratio * advantages).sum(dim=0).mean(dim=1)
            # Well I mean obviously if you do this there is no shaping because you can't differentiate through the inner update step...
            return regular_nl_loss, G_ts, values_loss


        return dice_loss, G_ts, values_loss


