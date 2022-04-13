import torch
import math
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
import argparse
import os
import datetime

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def checkpoint(agent1, agent2, info, tag, args):
    ckpt_dict = {
        "agent1": agent1,
        "agent2": agent2,
        "info": info
    }
    torch.save(ckpt_dict, os.path.join(args.save_dir, tag))

def load_from_checkpoint():
    assert args.load_path is not None
    print(f"loading model from {args.load_path}")
    ckpt_dict = torch.load(args.load_path)
    agent1 = ckpt_dict["agent1"]
    agent2 = ckpt_dict["agent2"]
    info = ckpt_dict["info"]
    return agent1, agent2, info


def print_info_on_sample_obs(sample_obs, th, vals):

    sample_obs = sample_obs.reshape(-1, 1, args.grid_size ** 2 * 4).to(device)

    # ONLY SUPPORTS 2 AGENTS
    # sample_obs = torch.stack((sample_obs, sample_obs))


    # print(sample_obs.shape)
    n_agents = 2

    # print(sample_obs.shape)

    h_p = torch.zeros(sample_obs.shape[-2],
                       args.hidden_size).to(device)
    h_v = torch.zeros(sample_obs.shape[-2],
                       args.hidden_size).to(device)

    for t in range(sample_obs.shape[0]):

        a1, lp1, v1, h_p1, h_v1, cat_act_probs1 = act(sample_obs[t], th, vals, h_p,
                                                      h_v)

        # policies, values, h_p, h_v = self.get_policy_vals_indices_for_iter(th, vals,
        #                                                       sample_obs[:,-1,:,:], h_p, h_v)
        # for i in range(n_agents):
            # print("Agent {}:".format(i + 1))
        print(cat_act_probs1)
        print(v1)





def print_policy_and_value_info(th, vals):

    if args.grid_size == 2:
        print("Simple One Step Example")
        sample_obs = torch.FloatTensor([[[0, 1],
                                         [0, 0]
                                         ],  # agent 1
                                        [[0, 0],
                                         [1, 0]
                                         ],  # agent 2
                                        [[1, 0],
                                         [0, 0]
                                         ],
                                        # we want agent 1 moving left and agent 2 moving right
                                        [[0, 0],
                                         [0, 1],
                                         ]]).reshape(1, args.grid_size ** 2 * 4)

        print_info_on_sample_obs(sample_obs, th, vals)

    elif args.grid_size == 3:
        # Policy test
        print("Simple One Step Example")
        sample_obs = torch.FloatTensor([[[0, 1, 0],
                                         [0, 0, 0],
                                         [0, 0, 0]],  # agent 1
                                        [[0, 0, 0],
                                         [1, 0, 0],
                                         [0, 0, 0]],  # agent 2
                                        [[1, 0, 0],
                                         [0, 0, 0],
                                         [0, 0, 0]],
                                        # we want agent 1 moving left and agent 2 moving right
                                        [[0, 0, 0],
                                         [0, 1, 0],
                                         [0, 0, 0]]]).reshape(1, 36)

        print_info_on_sample_obs(sample_obs, th, vals)

        print("Simple One Step Example 2")
        sample_obs = torch.FloatTensor([[[0, 0, 0],
                                         [0, 0, 0],
                                         [0, 0, 1]],  # agent 1
                                        [[0, 0, 0],
                                         [0, 1, 0],
                                         [0, 0, 0]],  # agent 2
                                        [[0, 0, 0],
                                         [0, 0, 1],
                                         [0, 0, 0]],
                                        # we want agent 1 moving up and agent 2 moving down
                                        [[0, 0, 0],
                                         [0, 0, 0],
                                         [0, 1, 0]]]).reshape(1, 36)

        print_info_on_sample_obs(sample_obs, th, vals)

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
                                           [0, 1, 0],
                                           [0, 0, 0]]]).reshape(1, 36)
        sample_obs_2 = torch.FloatTensor([[[0, 0, 0],
                                           [1, 0, 0],
                                           [0, 0, 0]],  # agent 1
                                          [[0, 0, 0],
                                           [0, 0, 0],
                                           [1, 0, 0]],  # agent 2
                                          [[0, 1, 0],
                                           [0, 0, 0],
                                           [0, 0, 0]],
                                          # red coin
                                          [[0, 0, 0],
                                           [0, 1, 0],
                                           [0, 0, 0]]]).reshape(1, 36)
        sample_obs = torch.stack((sample_obs_1, sample_obs_2), dim=1)

        print_info_on_sample_obs(sample_obs, th, vals)

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
                                           [0, 1, 0],
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
                                           [0, 1, 0]]]).reshape(1, 36)
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

        print_info_on_sample_obs(sample_obs, th, vals)

"""
OG COIN GAME.
COIN CANNOT SPAWN UNDER EITHER AGENT.
"""
class OGCoinGameGPU:
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

    def __init__(self, max_steps, batch_size, grid_size=3):
        self.max_steps = max_steps
        self.grid_size = grid_size
        self.batch_size = batch_size
        # The 4 channels stand for 2 players and 2 coin positions
        self.ob_space_shape = [4, grid_size, grid_size]
        self.NUM_STATES = np.prod(self.ob_space_shape)
        self.available_actions = 4
        self.step_count = None

    def reset(self):
        self.step_count = 0
        self.red_coin = torch.randint(2, size=(self.batch_size,)).to(device)

        red_pos_flat = torch.randint(self.grid_size * self.grid_size, size=(self.batch_size,)).to(device)
        self.red_pos = torch.stack((red_pos_flat // self.grid_size, red_pos_flat % self.grid_size), dim=-1)

        blue_pos_flat = (torch.randint(self.grid_size * self.grid_size - 1, size=(self.batch_size,)).to(device) + red_pos_flat) % (self.grid_size * self.grid_size)
        self.blue_pos = torch.stack((blue_pos_flat // self.grid_size, blue_pos_flat % self.grid_size), dim=-1)

        coin_pos_flat = torch.randint(self.grid_size * self.grid_size - 2, size=(self.batch_size,)).to(device)
        minpos = torch.min(red_pos_flat, blue_pos_flat)
        maxpos = torch.max(red_pos_flat, blue_pos_flat)
        coin_pos_flat[coin_pos_flat >= minpos] += 1
        coin_pos_flat[coin_pos_flat >= maxpos] += 1
        x = torch.logical_and(minpos == maxpos,
                          torch.randn_like(minpos.float()) < 1.0 / (
                                      self.grid_size * self.grid_size - 1))
        coin_pos_flat[x.long()] = minpos+1 # THIS IS TO FIX THE OFFSET BUG
        # coin_pos_flat[torch.logical_and(minpos==maxpos, torch.randn_like(minpos.float()) < 1.0 / (self.grid_size*self.grid_size - 1))] = minpos+1 # THIS IS TO FIX THE OFFSET BUG
        coin_pos_flat = coin_pos_flat % (self.grid_size * self.grid_size)
        self.coin_pos = torch.stack((coin_pos_flat // self.grid_size, coin_pos_flat % self.grid_size), dim=-1)

        # state = self._generate_state()
        # observations = [state, state]
        state = self._generate_state()
        state2 = state.clone()
        # print(state2.shape)
        state2[:, 0] = state[:, 1]
        state2[:, 1] = state[:, 0]
        state2[:, 2] = state[:, 3]
        state2[:, 3] = state[:, 2]
        observations = [state, state2]
        return observations

    def _generate_coins(self):

        mask = torch.logical_or(self._same_pos(self.coin_pos, self.blue_pos), self._same_pos(self.coin_pos, self.red_pos))
        self.red_coin = torch.where(mask, 1 - self.red_coin, self.red_coin)

        red_pos_flat = self.red_pos[mask,0] * self.grid_size + self.red_pos[mask, 1]
        blue_pos_flat = self.blue_pos[mask, 0] * self.grid_size + self.blue_pos[mask, 1]
        coin_pos_flat = torch.randint(self.grid_size * self.grid_size - 2, size=(self.batch_size,)).to(device)[mask]
        minpos = torch.min(red_pos_flat, blue_pos_flat)
        maxpos = torch.max(red_pos_flat, blue_pos_flat)
        coin_pos_flat[coin_pos_flat >= minpos] += 1
        coin_pos_flat[coin_pos_flat >= maxpos] += 1
        coin_pos_flat = coin_pos_flat % (self.grid_size * self.grid_size)
        self.coin_pos[mask] = torch.stack((coin_pos_flat // self.grid_size, coin_pos_flat % self.grid_size), dim=-1)

    def _same_pos(self, x, y):
        return torch.all(x == y, dim=-1)

    def _generate_state(self):
        red_pos_flat = self.red_pos[:,0] * self.grid_size + self.red_pos[:, 1]
        blue_pos_flat = self.blue_pos[:, 0] * self.grid_size + self.blue_pos[:, 1]

        # coin_pos_flat = self.coin_pos[:,0] * self.grid_size + self.coin_pos[:,1]
        coin_pos_flatter = self.coin_pos[:,0] * self.grid_size + self.coin_pos[:,1] + self.grid_size * self.grid_size * self.red_coin + 2 * self.grid_size * self.grid_size

        state = torch.zeros((self.batch_size, 4*self.grid_size*self.grid_size)).to(device)

        state.scatter_(1, coin_pos_flatter[:,None], 1)
        state = state.view((self.batch_size, 4, self.grid_size*self.grid_size))

        state[:,0].scatter_(1, red_pos_flat[:,None], 1)
        state[:,1].scatter_(1, blue_pos_flat[:,None], 1)

        return state.view(self.batch_size, 4, self.grid_size, self.grid_size)

    def step(self, actions):
        ac0, ac1 = actions

        self.step_count += 1

        self.red_pos = (self.red_pos + self.MOVES[ac0]) % self.grid_size
        self.blue_pos = (self.blue_pos + self.MOVES[ac1]) % self.grid_size

        # Compute rewards
        red_matches = self._same_pos(self.red_pos, self.coin_pos)
        red_reward = torch.zeros_like(self.red_coin).float()
        # red_reward[red_matches] = 1

        blue_matches = self._same_pos(self.blue_pos, self.coin_pos)
        blue_reward = torch.zeros_like(self.red_coin).float()
        # blue_reward[blue_matches] = 1

        red_reward[torch.logical_and(red_matches, self.red_coin)] = args.same_coin_reward
        blue_reward[torch.logical_and(blue_matches, 1 - self.red_coin)] = args.same_coin_reward
        red_reward[torch.logical_and(red_matches, 1 - self.red_coin)] = args.diff_coin_reward
        blue_reward[torch.logical_and(blue_matches, self.red_coin)] = args.diff_coin_reward

        red_reward[torch.logical_and(blue_matches, self.red_coin)] += (args.diff_coin_cost)
        blue_reward[torch.logical_and(red_matches, 1 - self.red_coin)] += (args.diff_coin_cost)
        # red_reward[torch.logical_and(blue_matches, self.red_coin)] -= 2
        # blue_reward[torch.logical_and(red_matches, 1 - self.red_coin)] -= 2

        total_rb_matches = torch.logical_and(red_matches, 1 - self.red_coin).float().mean()
        total_br_matches = torch.logical_and(blue_matches, self.red_coin).float().mean()

        total_rr_matches = red_matches.float().mean() - total_rb_matches
        total_bb_matches = blue_matches.float().mean() - total_br_matches

        self._generate_coins()
        reward = [red_reward.float(), blue_reward.float()]
        state = self._generate_state()
        state2 = state.clone()
        state2[:, 0] = state[:, 1]
        state2[:, 1] = state[:, 0]
        state2[:, 2] = state[:, 3]
        state2[:, 3] = state[:, 2]
        observations = [state, state2]
        # observations = [state, state]
        if self.step_count >= self.max_steps:
            done = torch.ones_like(self.red_coin)
        else:
            done = torch.zeros_like(self.red_coin)

        return observations, reward, done, (
        total_rr_matches, total_rb_matches,
        total_br_matches, total_bb_matches)


class CoinGameGPU:
    """
    Vectorized Coin Game environment.
    Note: slightly deviates from the Gym API.
    """
    NUM_AGENTS = 2
    NUM_ACTIONS = 4
    MOVES = torch.stack([
        torch.LongTensor([0, 1]), # right
        torch.LongTensor([0, -1]), # left
        torch.LongTensor([1, 0]), # down
        torch.LongTensor([-1, 0]), # up
    ], dim=0).to(device)

    def __init__(self, max_steps, batch_size, grid_size=3):
        self.max_steps = max_steps
        self.grid_size = grid_size
        self.batch_size = batch_size
        # The 4 channels stand for 2 players and 2 coin positions
        self.ob_space_shape = [4, grid_size, grid_size]
        self.NUM_STATES = np.prod(self.ob_space_shape)
        self.available_actions = 4
        self.step_count = None

    def reset(self):
        self.step_count = 0

        red_pos_flat = torch.randint(self.grid_size * self.grid_size,
                                     size=(self.batch_size,)).to(device)
        self.red_pos = torch.stack(
            (torch.div(red_pos_flat, self.grid_size, rounding_mode='floor') , red_pos_flat % self.grid_size),
            dim=-1)

        blue_pos_flat = torch.randint(self.grid_size * self.grid_size,
                                      size=(self.batch_size,)).to(device)
        self.blue_pos = torch.stack(
            (torch.div(blue_pos_flat, self.grid_size, rounding_mode='floor'), blue_pos_flat % self.grid_size),
            dim=-1)

        red_coin_pos_flat = torch.randint(self.grid_size * self.grid_size,
                                          size=(self.batch_size,)).to(device)
        blue_coin_pos_flat = torch.randint(self.grid_size * self.grid_size,
                                           size=(self.batch_size,)).to(device)

        self.red_coin_pos = torch.stack((torch.div(red_coin_pos_flat, self.grid_size, rounding_mode='floor'),
                                         red_coin_pos_flat % self.grid_size),
                                        dim=-1)
        self.blue_coin_pos = torch.stack((torch.div(blue_coin_pos_flat, self.grid_size, rounding_mode='floor'),
                                          blue_coin_pos_flat % self.grid_size),
                                         dim=-1)

        state = self._generate_state()
        state2 = state.clone()
        # print(state2.shape)
        state2[:,0] = state[:,1]
        state2[:,1] = state[:,0]
        state2[:,2] = state[:,3]
        state2[:,3] = state[:,2]
        observations = [state, state2]
        return observations

    def _generate_coins(self):
        mask_red = torch.logical_or(
            self._same_pos(self.red_coin_pos, self.blue_pos),
            self._same_pos(self.red_coin_pos, self.red_pos))
        red_coin_pos_flat = torch.randint(self.grid_size * self.grid_size,
                                          size=(self.batch_size,)).to(device)[
            mask_red]
        self.red_coin_pos[mask_red] = torch.stack((
                                                  torch.div(red_coin_pos_flat, self.grid_size, rounding_mode='floor'),
                                                  red_coin_pos_flat % self.grid_size),
                                                  dim=-1)

        mask_blue = torch.logical_or(
            self._same_pos(self.blue_coin_pos, self.blue_pos),
            self._same_pos(self.blue_coin_pos, self.red_pos))
        blue_coin_pos_flat = torch.randint(self.grid_size * self.grid_size,
                                           size=(self.batch_size,)).to(device)[
            mask_blue]
        self.blue_coin_pos[mask_blue] = torch.stack((
                                                    torch.div(blue_coin_pos_flat, self.grid_size, rounding_mode='floor'),
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

        state[:, 0].scatter_(1, red_pos_flat[:, None], 1)
        state[:, 1].scatter_(1, blue_pos_flat[:, None], 1)
        state[:, 2].scatter_(1, red_coin_pos_flat[:, None], 1)
        state[:, 3].scatter_(1, blue_coin_pos_flat[:, None], 1)

        return state.view(self.batch_size, 4, self.grid_size, self.grid_size)

    def step(self, actions):
        ac0, ac1 = actions

        self.step_count += 1

        self.red_pos = (self.red_pos + self.MOVES[ac0]) % self.grid_size
        self.blue_pos = (self.blue_pos + self.MOVES[ac1]) % self.grid_size

        # Compute rewards
        red_reward = torch.zeros(self.batch_size).to(device)
        red_red_matches = self._same_pos(self.red_pos, self.red_coin_pos)
        red_reward[red_red_matches] += args.same_coin_reward
        red_blue_matches = self._same_pos(self.red_pos, self.blue_coin_pos)
        red_reward[red_blue_matches] += args.diff_coin_reward

        blue_reward = torch.zeros(self.batch_size).to(device)
        blue_red_matches = self._same_pos(self.blue_pos, self.red_coin_pos)
        blue_reward[blue_red_matches] += args.diff_coin_reward
        blue_blue_matches = self._same_pos(self.blue_pos, self.blue_coin_pos)
        blue_reward[blue_blue_matches] += args.same_coin_reward

        red_reward[blue_red_matches] += args.diff_coin_cost # -= 2
        blue_reward[red_blue_matches] += args.diff_coin_cost # -= 2

        self._generate_coins()
        reward = [red_reward.float(), blue_reward.float()]
        state = self._generate_state()
        state2 = state.clone()
        state2[:, 0] = state[:, 1]
        state2[:, 1] = state[:, 0]
        state2[:, 2] = state[:, 3]
        state2[:, 3] = state[:, 2]
        observations = [state, state2]
        if self.step_count >= self.max_steps:
            done = torch.ones(self.batch_size).to(device)
        else:
            done = torch.zeros(self.batch_size).to(device)

        return observations, reward, done, (
        red_red_matches.float().mean(), red_blue_matches.float().mean(),
        blue_red_matches.float().mean(), blue_blue_matches.float().mean())



def magic_box(x):
    return torch.exp(x - x.detach())


class Memory():
    def __init__(self):
        self.self_logprobs = []
        self.other_logprobs = []
        self.values = []
        self.rewards = []

    def add(self, lp, other_lp, v, r):
        self.self_logprobs.append(lp)
        self.other_logprobs.append(other_lp)
        self.values.append(v)
        self.rewards.append(r)

    def dice_objective(self):
        self_logprobs = torch.stack(self.self_logprobs, dim=1)
        other_logprobs = torch.stack(self.other_logprobs, dim=1)
        values = torch.stack(self.values, dim=1)
        rewards = torch.stack(self.rewards, dim=1)

        # apply discount:
        cum_discount = torch.cumprod(args.gamma * torch.ones(*rewards.size()),
                                     dim=1).to(device) / args.gamma
        discounted_rewards = rewards * cum_discount
        discounted_values = values * cum_discount

        # stochastics nodes involved in rewards dependencies:
        dependencies = torch.cumsum(self_logprobs + other_logprobs, dim=1)

        # logprob of each stochastic nodes:
        stochastic_nodes = self_logprobs + other_logprobs

        # dice objective:
        dice_objective = torch.mean(
            torch.sum(magic_box(dependencies) * discounted_rewards, dim=1))

        if use_baseline:
            # variance_reduction:
            baseline_term = torch.mean(
                torch.sum((1 - magic_box(stochastic_nodes)) * discounted_values,
                          dim=1))
            dice_objective = dice_objective + baseline_term

        return -dice_objective  # want to minimize -objective

    def value_loss(self):
        values = torch.stack(self.values, dim=1)
        rewards = torch.stack(self.rewards, dim=1)
        return torch.mean((rewards - values) ** 2)


def apply(batch_states, theta, hidden):
    #     import pdb; pdb.set_trace()
    batch_states = batch_states.flatten(start_dim=1)
    x = batch_states.matmul(theta[0])
    x = theta[1] + x

    x = torch.relu(x)

    gate_x = x.matmul(theta[2])
    gate_x = gate_x + theta[3]

    gate_h = hidden.matmul(theta[4])
    gate_h = gate_h + theta[5]

    #     gate_x = gate_x.squeeze()
    #     gate_h = gate_h.squeeze()

    i_r, i_i, i_n = gate_x.chunk(3, 1)
    h_r, h_i, h_n = gate_h.chunk(3, 1)

    resetgate = torch.sigmoid(i_r + h_r)
    inputgate = torch.sigmoid(i_i + h_i)
    newgate = torch.tanh(i_n + (resetgate * h_n))

    hy = newgate + inputgate * (hidden - newgate)

    out = hy.matmul(theta[6])
    out = out + theta[7]

    return hy, out


def act(batch_states, theta_p, theta_v, h_p, h_v):
    h_p, out = apply(batch_states, theta_p, h_p)
    categorical_act_probs = torch.softmax(out, dim=-1)
    h_v, values = apply(batch_states, theta_v, h_v)
    dist = Categorical(categorical_act_probs)
    actions = dist.sample()
    log_probs_actions = dist.log_prob(actions)
    return actions, log_probs_actions, values.squeeze(-1), h_p, h_v, categorical_act_probs


def get_gradient(objective, theta):
    # create differentiable gradient for 2nd orders:
    grad_objective = torch.autograd.grad(objective, (theta), create_graph=True)
    return grad_objective


def step(theta1, theta2, values1, values2):
    # just to evaluate progress:
    (s1, s2) = env.reset()
    score1 = 0
    score2 = 0
    h_p1, h_v1, h_p2, h_v2 = (
    torch.zeros(args.batch_size, args.hidden_size).to(device),
    torch.zeros(args.batch_size, args.hidden_size).to(device),
    torch.zeros(args.batch_size, args.hidden_size).to(device),
    torch.zeros(args.batch_size, args.hidden_size).to(device))
    rr_matches_record, rb_matches_record, br_matches_record, bb_matches_record = 0., 0., 0., 0.

    for t in range(args.len_rollout):
        a1, lp1, v1, h_p1, h_v1, cat_act_probs1 = act(s1, theta1, values1, h_p1, h_v1)
        a2, lp2, v2, h_p2, h_v2, cat_act_probs2 = act(s2, theta2, values2, h_p2, h_v2)
        (s1, s2), (r1, r2), _, info = env.step((a1, a2))
        # cumulate scores
        score1 += torch.mean(r1) / float(args.len_rollout)
        score2 += torch.mean(r2) / float(args.len_rollout)
        # print(info)
        rr_matches, rb_matches, br_matches, bb_matches = info
        rr_matches_record += rr_matches
        rb_matches_record += rb_matches
        br_matches_record += br_matches
        bb_matches_record += bb_matches

    return (score1, score2), (rr_matches_record, rb_matches_record, br_matches_record, bb_matches_record)


class Agent():
    def __init__(self, input_size, hidden_size, action_size):
        self.hidden_size = hidden_size
        self.theta_p = nn.ParameterList([
            # Linear 1
            nn.Parameter(
                torch.zeros((input_size, hidden_size * 3), requires_grad=True)),
            nn.Parameter(torch.zeros(hidden_size * 3, requires_grad=True)),

            # x2h GRU
            nn.Parameter(torch.zeros((hidden_size * 3, hidden_size * 3),
                                     requires_grad=True)),
            nn.Parameter(torch.zeros(hidden_size * 3, requires_grad=True)),

            # h2h GRU
            nn.Parameter(torch.zeros((hidden_size, hidden_size * 3),
                                     requires_grad=True)),
            nn.Parameter(torch.zeros(hidden_size * 3, requires_grad=True)),

            # Linear 2
            nn.Parameter(
                torch.zeros((hidden_size, action_size), requires_grad=True)),
            nn.Parameter(torch.zeros(action_size, requires_grad=True)),
        ]).to(device)

        self.theta_v = nn.ParameterList([
            # Linear 1
            nn.Parameter(
                torch.zeros((input_size, hidden_size * 3), requires_grad=True)),
            nn.Parameter(torch.zeros(hidden_size * 3, requires_grad=True)),

            # x2h GRU
            nn.Parameter(torch.zeros((hidden_size * 3, hidden_size * 3),
                                     requires_grad=True)),
            nn.Parameter(torch.zeros(hidden_size * 3, requires_grad=True)),

            # h2h GRU
            nn.Parameter(torch.zeros((hidden_size, hidden_size * 3),
                                     requires_grad=True)),
            nn.Parameter(torch.zeros(hidden_size * 3, requires_grad=True)),

            # Linear 2
            nn.Parameter(torch.zeros((hidden_size, 1), requires_grad=True)),
            nn.Parameter(torch.zeros(1, requires_grad=True)),
        ]).to(device)

        self.reset_parameters()
        if args.optim.lower() == 'adam':
            self.theta_optimizer = torch.optim.Adam(self.theta_p, lr=args.lr_out)
            self.value_optimizer = torch.optim.Adam(self.theta_v, lr=args.lr_v)
        elif args.optim.lower() == 'sgd':
            self.theta_optimizer = torch.optim.SGD(self.theta_p, lr=args.lr_out)
            self.value_optimizer = torch.optim.SGD(self.theta_v, lr=args.lr_v)
        else:
            raise Exception("Unknown or Not Implemented Optimizer")

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.theta_p:
            w.data.uniform_(-std, std)

        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.theta_v:
            w.data.uniform_(-std, std)

    def theta_update(self, objective):
        self.theta_optimizer.zero_grad()
        objective.backward(retain_graph=True)
        self.theta_optimizer.step()

    def value_update(self, loss):
        self.value_optimizer.zero_grad()
        loss.backward()
        self.value_optimizer.step()

    def get_policies_for_states(self):
        h_p1, h_v1 = (
            torch.zeros(args.batch_size, self.hidden_size).to(device),
            torch.zeros(args.batch_size, self.hidden_size).to(device))

        cat_act_probs = []

        for t in range(args.len_rollout):
            s1 = self.state_history[t]
            a1, lp1, v1, h_p1, h_v1, cat_act_probs1 = act(s1, self.theta_p,
                                                          self.theta_v, h_p1,
                                                          h_v1)
            cat_act_probs.append(cat_act_probs1)

        return torch.stack(cat_act_probs, dim=1)

    def get_other_policies_for_states(self, other_theta, other_values, state_history):
        # Perhaps really should not be named 1, but whatever.
        h_p1, h_v1 = (
            torch.zeros(args.batch_size, self.hidden_size).to(device),
            torch.zeros(args.batch_size, self.hidden_size).to(device))

        cat_act_probs = []

        for t in range(args.len_rollout):
            s1 = state_history[t]
            a1, lp1, v1, h_p1, h_v1, cat_act_probs1 = act(s1, other_theta,
                                                          other_values, h_p1,
                                                          h_v1)
            cat_act_probs.append(cat_act_probs1)

        return torch.stack(cat_act_probs, dim=1)

    def in_lookahead(self, other_theta, other_values, first_inner_step=False):
        (s1, s2) = env.reset()
        other_memory = Memory()
        h_p1, h_v1, h_p2, h_v2 = (
        torch.zeros(args.batch_size, self.hidden_size).to(device),
        torch.zeros(args.batch_size, self.hidden_size).to(device),
        torch.zeros(args.batch_size, self.hidden_size).to(device),
        torch.zeros(args.batch_size, self.hidden_size).to(device))

        if first_inner_step:
            cat_act_probs_other = []
            other_state_history = []
            other_state_history.append(s2)

        for t in range(args.len_rollout):
            a1, lp1, v1, h_p1, h_v1, cat_act_probs1 = act(s1, self.theta_p, self.theta_v, h_p1,
                                          h_v1)
            a2, lp2, v2, h_p2, h_v2, cat_act_probs2 = act(s2, other_theta, other_values, h_p2,
                                          h_v2)
            (s1, s2), (r1, r2), _, _ = env.step((a1, a2))
            other_memory.add(lp2, lp1, v2, r2)
            if first_inner_step:
                cat_act_probs_other.append(cat_act_probs2)
                other_state_history.append(s2)

        if not first_inner_step:
            curr_pol_probs = self.get_other_policies_for_states(other_theta, other_values, self.other_state_history)
            kl_div = torch.nn.functional.kl_div(torch.log(curr_pol_probs), self.ref_cat_act_probs_other.detach(), log_target=False, reduction='batchmean')
            # print(curr_pol_probs.shape)
            print(kl_div)

        other_objective = other_memory.dice_objective()
        if not first_inner_step:
            other_objective += args.inner_beta * kl_div # we want to min kl div

        grad = get_gradient(other_objective, other_theta)

        if first_inner_step:
            self.ref_cat_act_probs_other = torch.stack(cat_act_probs_other, dim=1)
            self.other_state_history = torch.stack(other_state_history, dim=0)

        return grad

    def out_lookahead(self, other_theta, other_values, first_outer_step=False):
        (s1, s2) = env.reset()
        memory = Memory()
        h_p1, h_v1, h_p2, h_v2 = (
        torch.zeros(args.batch_size, self.hidden_size).to(device),
        torch.zeros(args.batch_size, self.hidden_size).to(device),
        torch.zeros(args.batch_size, self.hidden_size).to(device),
        torch.zeros(args.batch_size, self.hidden_size).to(device))
        if first_outer_step:
            cat_act_probs_self = []
            state_history = []
            state_history.append(s1)
        if args.ent_reg > 0:
            ent_vals = []
        for t in range(args.len_rollout):
            a1, lp1, v1, h_p1, h_v1, cat_act_probs1 = act(s1, self.theta_p, self.theta_v, h_p1,
                                          h_v1)
            a2, lp2, v2, h_p2, h_v2, cat_act_probs2 = act(s2, other_theta, other_values, h_p2,
                                          h_v2)
            (s1, s2), (r1, r2), _, _ = env.step((a1, a2))
            memory.add(lp1, lp2, v1, r1)
            if first_outer_step:
                cat_act_probs_self.append(cat_act_probs1)
                state_history.append(s1)
            if args.ent_reg > 0:
                ent_vals.append(cat_act_probs1)

        if not first_outer_step:
            curr_pol_probs = self.get_policies_for_states()
            kl_div = torch.nn.functional.kl_div(torch.log(curr_pol_probs), self.ref_cat_act_probs.detach(), log_target=False, reduction='batchmean')
            # print(curr_pol_probs.shape)
            print(kl_div)
            # kl_div2 = (curr_pol_probs * torch.log(curr_pol_probs / self.ref_cat_act_probs.detach())).sum() / self.batch_size
            # print(kl_div2)

        # update self theta
        objective = memory.dice_objective()
        if not first_outer_step:
            objective += args.outer_beta * kl_div # we want to min kl div
        if args.ent_reg > 0:
            ent_vals = torch.stack(ent_vals, dim=0)
            ent_calc = - (ent_vals * torch.log(ent_vals)).sum(dim=-1).mean()
            # print(ent_calc)
            objective += -ent_calc * args.ent_reg # but we want to max entropy (min negative entropy)
        self.theta_update(objective)
        # update self value:
        v_loss = memory.value_loss()
        self.value_update(v_loss)

        if first_outer_step:
            self.ref_cat_act_probs = torch.stack(cat_act_probs_self, dim=1)
            self.state_history = torch.stack(state_history, dim=0)
            # return torch.stack(cat_act_probs_self, dim=1), torch.stack(state_history, dim=1)

def play(agent1, agent2, n_lookaheads, outer_steps):
    joint_scores = []
    print("start iterations with", n_lookaheads, "inner steps and", outer_steps, "outer steps:")
    same_colour_coins_record = []
    diff_colour_coins_record = []
    coins_collected_info = (same_colour_coins_record, diff_colour_coins_record)
    for update in range(args.n_update):

        start_theta1 = [tp.detach().clone().requires_grad_(True) for tp in
                            agent1.theta_p]
        start_val1 = [tv.detach().clone().requires_grad_(True) for tv in
                        agent1.theta_v]
        start_theta2 = [tp.detach().clone().requires_grad_(True) for tp in
                            agent2.theta_p]
        start_val2 = [tv.detach().clone().requires_grad_(True) for tv in
                        agent2.theta_v]

        for outer_step in range(outer_steps):
            # copy other's parameters:
            theta2_ = [tp.detach().clone().requires_grad_(True) for tp in
                       start_theta2]
            values2_ = [tv.detach().clone().requires_grad_(True) for tv in
                        start_val2]

            for inner_step in range(n_lookaheads):
                if inner_step == 0:
                    # estimate other's gradients from in_lookahead:
                    grad2 = agent1.in_lookahead(theta2_, values2_, first_inner_step=True)
                else:
                    grad2 = agent1.in_lookahead(theta2_, values2_, first_inner_step=False)
                # update other's theta
                theta2_ = [theta2_[i] - args.lr_in * grad2[i] for i in
                           range(len(theta2_))]

            # update own parameters from out_lookahead:
            if outer_step == 0:
                agent1.out_lookahead(theta2_, values2_, first_outer_step=True)
            else:
                agent1.out_lookahead(theta2_, values2_, first_outer_step=False)


        for outer_step in range(outer_steps):
            theta1_ = [tp.detach().clone().requires_grad_(True) for tp in
                       start_theta1]
            values1_ = [tv.detach().clone().requires_grad_(True) for tv in
                        start_val1]

            for inner_step in range(n_lookaheads):
                # estimate other's gradients from in_lookahead:
                if inner_step == 0:
                    grad1 = agent2.in_lookahead(theta1_, values1_, first_inner_step=True)
                else:
                    grad1 = agent2.in_lookahead(theta1_, values1_, first_inner_step=False)
                # update other's theta
                theta1_ = [theta1_[i] - args.lr_in * grad1[i] for i in
                           range(len(theta1_))]

            if outer_step == 0:
                agent2.out_lookahead(theta1_, values1_, first_outer_step=True)
            else:
                agent2.out_lookahead(theta1_, values1_, first_outer_step=False)

        # evaluate progress:
        score, info = step(agent1.theta_p, agent2.theta_p, agent1.theta_v,
                           agent2.theta_v)
        rr_matches, rb_matches, br_matches, bb_matches = info
        same_colour_coins = (rr_matches + bb_matches).item()
        diff_colour_coins = (rb_matches + br_matches).item()
        same_colour_coins_record.append(same_colour_coins)
        diff_colour_coins_record.append(diff_colour_coins)
        joint_scores.append(0.5 * (score[0] + score[1]))

        # print
        if update % args.print_every == 0:
            #             p1 = [p.item() for p in torch.sigmoid(agent1.theta)]
            #             p2 = [p.item() for p in torch.sigmoid(agent2.theta)]
            #             print('update', update, 'score (%.3f,%.3f)' % (score[0], score[1]) , 'policy (agent1) = {%.3f, %.3f, %.3f, %.3f, %.3f}' % (p1[0], p1[1], p1[2], p1[3], p1[4]),' (agent2) = {%.3f, %.3f, %.3f, %.3f, %.3f}' % (p2[0], p2[1], p2[2], p2[3], p2[4]))
            print("*" * 10)
            print("Epoch: {}".format(update + 1))
            print(f"Score 0: {score[0]}")
            print(f"Score 1: {score[1]}")
            print("Same coins: {}".format(same_colour_coins))
            print("Diff coins: {}".format(diff_colour_coins))
            print("RR coins {}".format(rr_matches))
            print("RB coins {}".format(rb_matches))
            print("BR coins {}".format(br_matches))
            print("BB coins {}".format(bb_matches))
            print("Agent 1 Sample Obs Info:")
            print_policy_and_value_info(agent1.theta_p, agent1.theta_v)
            print("Agent 2 Sample Obs Info:")
            print_policy_and_value_info(agent2.theta_p, agent2.theta_v)

        if update % args.checkpoint_every == 0:
            now = datetime.datetime.now()
            checkpoint(agent1, agent2, coins_collected_info,
                       "checkpoint_{}_{}.pt".format(update + 1, now.strftime(
                           '%Y-%m-%d_%H-%M')), args)

    return joint_scores




if __name__ == "__main__":
    parser = argparse.ArgumentParser("NPLOLA")
    parser.add_argument("--inner_steps", type=int, default=1, help="inner loop steps for DiCE")
    parser.add_argument("--outer_steps", type=int, default=1, help="outer loop steps for POLA")
    parser.add_argument("--lr_out", type=float, default=0.005,
                        help="outer loop learning rate: same learning rate across all policies for now")
    parser.add_argument("--lr_in", type=float, default=0.05,
                        help="inner loop learning rate (eta): this has no use in the naive learning case. Used for the gradient step done for the lookahead for other agents during LOLA (therefore, often scaled to be higher than the outer learning rate in non-proximal LOLA). Note that this has a different meaning for the Taylor approx vs. actual update versions. A value of eta=1 is perfectly reasonable for the Taylor approx version as this balances the scale of the gradient with the naive learning term (and will be multiplied by the outer learning rate after), whereas for the actual update version with neural net, 1 is way too big an inner learning rate. For prox, this is the learning rate on the inner prox loop so is not that important - you want big enough to be fast-ish, but small enough to converge.")
    parser.add_argument("--lr_v", type=float, default=0.001,
                        help="same learning rate across all policies for now. Should be around maybe 0.001 or less for neural nets to avoid instability")
    parser.add_argument("--gamma", type=float, default=0.96, help="discount rate")
    parser.add_argument("--n_update", type=int, default=5000, help="number of epochs to run")
    parser.add_argument("--len_rollout", type=int, default=50, help="How long we want the time horizon of the game to be (number of steps before termination/number of iterations of the IPD)")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1, help="for seed")
    parser.add_argument("--hidden_size", type=int, default=32)
    parser.add_argument("--print_every", type=int, default=1, help="Print every x number of epochs")
    parser.add_argument("--outer_beta", type=float, default=0.0, help="for outer kl penalty with POLA")
    parser.add_argument("--inner_beta", type=float, default=0.0, help="for inner kl penalty with POLA")
    parser.add_argument("--save_dir", type=str, default='./checkpoints')
    parser.add_argument("--checkpoint_every", type=int, default=1000, help="Epochs between checkpoint save")
    parser.add_argument("--load_path", type=str, default=None, help="Give path if loading from a checkpoint")
    parser.add_argument("--ent_reg", type=float, default=0.0, help="entropy regularizer")
    parser.add_argument("--diff_coin_reward", type=float, default=1.0, help="changes problem setting (the reward for picking up coin of different colour)")
    parser.add_argument("--diff_coin_cost", type=float, default=-2.0, help="changes problem setting (the cost to the opponent when you pick up a coin of their colour)")
    parser.add_argument("--same_coin_reward", type=float, default=1.0, help="changes problem setting (the reward for picking up coin of same colour)")
    parser.add_argument("--grid_size", type=int, default=3)
    parser.add_argument("--og_coin_game", action="store_true", help="use the original coin game formulation")
    parser.add_argument("--optim", type=str, default="adam")


    use_baseline = True

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    input_size = args.grid_size ** 2 * 4
    action_size = 4

    if args.og_coin_game:
        env = OGCoinGameGPU(max_steps=args.len_rollout, batch_size=args.batch_size, grid_size=args.grid_size)

    else:
        env = CoinGameGPU(max_steps=args.len_rollout, batch_size=args.batch_size, grid_size=args.grid_size)

    if args.load_path is None:
        agent1 = Agent(input_size, args.hidden_size, action_size)
        agent2 = Agent(input_size, args.hidden_size, action_size)
    else:
        agent1, agent2, coins_collected_info = load_from_checkpoint()
        print(coins_collected_info)

    scores = play(agent1, agent2, args.inner_steps, args.outer_steps)
