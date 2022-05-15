# Adapted from https://github.com/alexis-jacq/LOLA_DiCE/blob/master/ipd_DiCE.py

import torch
import math
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
import argparse
import os
import datetime

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def checkpoint(agent1, agent2, info, scores, vs_fixed_scores, tag, args):
    ckpt_dict = {
        "agent1": agent1,
        "agent2": agent2,
        "info": info,
        "scores": scores,
        "vs_fixed_scores": vs_fixed_scores
    }
    torch.save(ckpt_dict, os.path.join(args.save_dir, tag))

def load_from_checkpoint():
    assert args.load_path is not None
    print(f"loading model from {args.load_path}")
    ckpt_dict = torch.load(args.load_path)
    agent1 = ckpt_dict["agent1"]
    agent2 = ckpt_dict["agent2"]
    info = ckpt_dict["info"]
    scores = ckpt_dict["scores"]
    vs_fixed_scores = ckpt_dict["vs_fixed_scores"]
    return agent1, agent2, info, scores, vs_fixed_scores

def reverse_cumsum(x, dim):
    return x + torch.sum(x, dim=dim, keepdims=True) - torch.cumsum(x, dim=dim)

def print_info_on_sample_obs(sample_obs, th, vals):
    sample_obs = sample_obs.reshape(-1, 1, input_size).to(device)

    h_p = torch.zeros(sample_obs.shape[-2],
                       args.hidden_size).to(device)
    h_v = torch.zeros(sample_obs.shape[-2],
                       args.hidden_size).to(device)

    for t in range(sample_obs.shape[0]):

        a1, lp1, v1, h_p1, h_v1, cat_act_probs1 = act(sample_obs[t], th, vals, h_p,
                                                      h_v)

        print(cat_act_probs1)
        print(v1)


def print_policy_and_value_info(th, vals):
    if args.env == "ipd":
        print("Simple One Step Examples")
        print("Start")
        sample_obs = torch.FloatTensor([[[0, 0, 1],
                                         [0, 0, 1]]]).reshape(1, input_size)
        print_info_on_sample_obs(sample_obs, th, vals)
        print("DD") # NOTE these are from the perspective of: (my past action, opp past action)
        # Not (p1 action, p2 action) as I did in my old file (or in the LOLA_exact file)
        sample_obs = torch.FloatTensor([[[1, 0, 0],
                                         [1, 0, 0]]]).reshape(1, input_size)
        print_info_on_sample_obs(sample_obs, th, vals)
        print("DC")
        sample_obs = torch.FloatTensor([[[1, 0, 0],
                                         [0, 1, 0]]]).reshape(1, input_size)
        print_info_on_sample_obs(sample_obs, th, vals)
        print("CD")
        sample_obs = torch.FloatTensor([[[0, 1, 0],
                                         [1, 0, 0]]]).reshape(1, input_size)
        print_info_on_sample_obs(sample_obs, th, vals)
        print("CC")
        sample_obs = torch.FloatTensor([[[0, 1, 0],
                                         [0, 1, 0]]]).reshape(1, input_size)
        print_info_on_sample_obs(sample_obs, th, vals)

    elif args.env == "ogcoin":
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
                                             ]]).reshape(1, input_size)

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
                                             [0, 0, 0]]]).reshape(1, input_size)

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
                                             [0, 1, 0]]]).reshape(1, input_size)

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
                                               [0, 0, 0]]]).reshape(1, input_size)
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
                                               [0, 0, 0]]]).reshape(1, input_size)
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
                                               [0, 0, 0]]]).reshape(1, input_size)
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
                                               [0, 1, 0]]]).reshape(1, input_size)
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
                                               [0, 1, 0]]]).reshape(1, input_size)
            # Want to see prob of going right going down.
            sample_obs = torch.stack((sample_obs_1, sample_obs_2, sample_obs_3),
                                     dim=1)

            print_info_on_sample_obs(sample_obs, th, vals)



# Adapted from https://github.com/alexis-jacq/LOLA_DiCE/blob/master/envs/prisoners_dilemma.py
class IteratedPrisonersDilemma:
    """
    A two-agent vectorized environment.
    Possible actions for each agent are (C)ooperate and (D)efect.
    """
    # Possible actions
    NUM_AGENTS = 2
    NUM_ACTIONS = 2
    NUM_STATES = 5
    ONE_HOT_REPR_DIM = 3

    def __init__(self, max_steps, batch_size=1):
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.payout_mat = torch.FloatTensor([[-2,0],[-3,-1]]).to(device)
        # One hot state representation because this would scale to n agents
        self.states = torch.FloatTensor([[[[1, 0, 0], [1, 0, 0]], #DD (first state is what I last did, second state is what opp last did)
                                          [[1, 0, 0], [0, 1, 0]]], #DC
                                         [[[0, 1, 0], [1, 0, 0]], #CD
                                          [[0, 1, 0], [0, 1, 0]]]]).to(device) #CC
        if args.init_state_coop:
            self.init_state = torch.FloatTensor([[0, 1, 0], [0, 1, 0]]).to(device)
        else:
            self.init_state = torch.FloatTensor([[0, 0, 1], [0, 0, 1]]).to(device)
        self.step_count = None

    def reset(self):
        self.step_count = 0
        init_state = self.init_state.repeat(self.batch_size, 1, 1)
        observation = [init_state, init_state]
        return observation

    def step(self, action):
        ac0, ac1 = action
        self.step_count += 1

        r0 = self.payout_mat[ac0, ac1]
        r1 = self.payout_mat[ac1, ac0]
        s0 = self.states[ac0, ac1]
        s1 = self.states[ac1, ac0]
        observation = [s0, s1]
        reward = [r0, r1]
        done = (self.step_count == self.max_steps)
        return observation, reward, done, None



"""
OG COIN GAME (But with bugs fixed, also vectorized).
Coin cannot spawn under any agent
Note that both agents can occupy the same spot on the grid
If both agents collect a coin at the same time, they both get the rewards associated
with the collection. To split the rewards (as if taking an expectation where the 
coin is randomly allocated to one of the agents), use --split_coins
"""
class OGCoinGameGPU:
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
        self.red_coin = torch.randint(2, size=(self.batch_size,)).to(device)

        red_pos_flat = torch.randint(self.grid_size * self.grid_size, size=(self.batch_size,)).to(device)
        self.red_pos = torch.stack((red_pos_flat // self.grid_size, red_pos_flat % self.grid_size), dim=-1)

        blue_pos_flat = torch.randint(self.grid_size * self.grid_size, size=(self.batch_size,)).to(device)
        self.blue_pos = torch.stack((blue_pos_flat // self.grid_size, blue_pos_flat % self.grid_size), dim=-1)

        coin_pos_flat = torch.randint(self.grid_size * self.grid_size - 2, size=(self.batch_size,)).to(device)
        minpos = torch.min(red_pos_flat, blue_pos_flat)
        maxpos = torch.max(red_pos_flat, blue_pos_flat)
        coin_pos_flat[coin_pos_flat >= minpos] += 1
        coin_pos_flat[coin_pos_flat >= maxpos] += 1

        # Regenerate coins when both agents are on the same spot, use a uniform
        # distribution among the 8 other possible spots
        same_agents_pos = (minpos == maxpos)
        coin_pos_flat[same_agents_pos] = torch.randint(self.grid_size * self.grid_size - 1, size=(same_agents_pos.sum(),)).to(device) + 1 + minpos[same_agents_pos]

        coin_pos_flat = coin_pos_flat % (self.grid_size * self.grid_size)

        # Test distribution of coins
        # print((minpos == maxpos).sum())
        # for i in range(self.grid_size * self.grid_size):
        #     print(torch.logical_and(minpos == maxpos, (coin_pos_flat == (minpos + i) % (self.grid_size * self.grid_size)) ).sum())
        assert (coin_pos_flat == red_pos_flat).sum() == 0
        assert (coin_pos_flat == blue_pos_flat).sum() == 0

        self.coin_pos = torch.stack((coin_pos_flat // self.grid_size, coin_pos_flat % self.grid_size), dim=-1)

        state = self._generate_state()
        state2 = state.clone()
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

        same_agents_pos = (minpos == maxpos)

        # Regenerate coins when both agents are on the same spot, regenerate uniform among the 8 other possible spots
        coin_pos_flat[same_agents_pos] = torch.randint(
            self.grid_size * self.grid_size - 1,
            size=(same_agents_pos.sum(),)).to(device) + 1 + minpos[same_agents_pos]

        coin_pos_flat = coin_pos_flat % (self.grid_size * self.grid_size)

        # Test distribution of coins
        # print((minpos == maxpos).sum())
        # for i in range(self.grid_size * self.grid_size):
        #     print(torch.logical_and(minpos == maxpos, (coin_pos_flat == (minpos + i) % (self.grid_size * self.grid_size)) ).sum())

        self.coin_pos[mask] = torch.stack((coin_pos_flat // self.grid_size, coin_pos_flat % self.grid_size), dim=-1)

    def _same_pos(self, x, y):
        return torch.all(x == y, dim=-1)

    def _generate_state(self):
        red_pos_flat = self.red_pos[:,0] * self.grid_size + self.red_pos[:, 1]
        blue_pos_flat = self.blue_pos[:, 0] * self.grid_size + self.blue_pos[:, 1]

        # 1 - self.red_coin here in order to have the red coin show up as obs in the second to last, rather than last of the 4 dimensions
        coin_pos_flatter = self.coin_pos[:,0] * self.grid_size + self.coin_pos[:,1] + self.grid_size * self.grid_size * (1-self.red_coin) + 2 * self.grid_size * self.grid_size

        state = torch.zeros((self.batch_size, 4*self.grid_size*self.grid_size)).to(device)

        state.scatter_(1, coin_pos_flatter[:,None], 1)
        state = state.view((self.batch_size, 4, self.grid_size*self.grid_size))

        state[:,0].scatter_(1, red_pos_flat[:,None], 1)
        state[:,1].scatter_(1, blue_pos_flat[:,None], 1)

        state = state.view(self.batch_size, 4, self.grid_size, self.grid_size)

        return state

    def step(self, actions):
        ac0, ac1 = actions

        self.step_count += 1

        self.red_pos = (self.red_pos + self.MOVES[ac0]) % self.grid_size
        self.blue_pos = (self.blue_pos + self.MOVES[ac1]) % self.grid_size

        # Compute rewards
        red_matches = self._same_pos(self.red_pos, self.coin_pos)
        red_reward = torch.zeros_like(self.red_coin).float()

        blue_matches = self._same_pos(self.blue_pos, self.coin_pos)
        blue_reward = torch.zeros_like(self.red_coin).float()

        red_reward[torch.logical_and(red_matches, self.red_coin)] = args.same_coin_reward
        blue_reward[torch.logical_and(blue_matches, 1 - self.red_coin)] = args.same_coin_reward
        red_reward[torch.logical_and(red_matches, 1 - self.red_coin)] = args.diff_coin_reward
        blue_reward[torch.logical_and(blue_matches, self.red_coin)] = args.diff_coin_reward

        red_reward[torch.logical_and(blue_matches, self.red_coin)] += args.diff_coin_cost
        blue_reward[torch.logical_and(red_matches, 1 - self.red_coin)] += args.diff_coin_cost

        if args.split_coins:
            both_matches = torch.logical_and(self._same_pos(self.red_pos, self.coin_pos), self._same_pos(self.blue_pos, self.coin_pos))
            red_reward[both_matches] *= 0.5
            blue_reward[both_matches] *= 0.5

        total_rb_matches = torch.logical_and(red_matches, 1 - self.red_coin).float().mean()
        total_br_matches = torch.logical_and(blue_matches, self.red_coin).float().mean()

        total_rr_matches = red_matches.float().mean() - total_rb_matches
        total_bb_matches = blue_matches.float().mean() - total_br_matches

        self._generate_coins()
        reward = [red_reward.float(), blue_reward.float()]
        state = self._generate_state()
        state2 = state.clone()
        # Because each agent sees the obs as if they are the "main" or "red" agent.
        # This is to be consistent with the self-centric IPD formulation too.
        state2[:, 0] = state[:, 1]
        state2[:, 1] = state[:, 0]
        state2[:, 2] = state[:, 3]
        state2[:, 3] = state[:, 2]
        observations = [state, state2]
        if self.step_count >= self.max_steps:
            done = torch.ones_like(self.red_coin)
        else:
            done = torch.zeros_like(self.red_coin)

        return observations, reward, done, (
        total_rr_matches, total_rb_matches,
        total_br_matches, total_bb_matches)

    def get_moves_shortest_path_to_coin(self, red_agent_perspective=True):
        # Ties broken arbitrarily, in this case, since I check the vertical distance later
        # priority is given to closing vertical distance (making up or down moves)
        # before horizontal moves
        if red_agent_perspective:
            agent_pos = self.red_pos
        else:
            agent_pos = self.blue_pos
        actions = torch.zeros(self.batch_size) - 1
        # assumes red agent perspective
        horiz_dist_right = (self.coin_pos[:, 1] - agent_pos[:, 1]) % self.grid_size
        horiz_dist_left = (agent_pos[:, 1] - self.coin_pos[:, 1]) % self.grid_size

        vert_dist_down = (self.coin_pos[:, 0] - agent_pos[:,
                                                0]) % self.grid_size
        vert_dist_up = (agent_pos[:, 0] - self.coin_pos[:,
                                             0]) % self.grid_size
        actions[horiz_dist_right < horiz_dist_left] = 0
        actions[horiz_dist_left < horiz_dist_right] = 1
        actions[vert_dist_down < vert_dist_up] = 2
        actions[vert_dist_up < vert_dist_down] = 3
        # Assumes no coin spawns under agent
        assert torch.logical_and(horiz_dist_right == horiz_dist_left, vert_dist_down == vert_dist_up).sum() == 0

        return actions.long()

    def get_moves_away_from_coin(self, moves_towards_coin):
        opposite_moves = torch.zeros_like(moves_towards_coin)
        opposite_moves[moves_towards_coin == 0] = 1
        opposite_moves[moves_towards_coin == 1] = 0
        opposite_moves[moves_towards_coin == 2] = 3
        opposite_moves[moves_towards_coin == 3] = 2
        return opposite_moves

    def get_coop_action(self, red_agent_perspective=True):
        # move toward coin if same colour, away if opposite colour
        # An agent that always does this is considered to 'always cooperate'
        moves_towards_coin = self.get_moves_shortest_path_to_coin(red_agent_perspective=red_agent_perspective)
        moves_away_from_coin = self.get_moves_away_from_coin(moves_towards_coin)
        coop_moves = torch.zeros_like(moves_towards_coin) - 1
        if red_agent_perspective:
            is_my_coin = self.red_coin
        else:
            is_my_coin = 1 - self.red_coin

        coop_moves[is_my_coin == 1] = moves_towards_coin[is_my_coin == 1]
        coop_moves[is_my_coin == 0] = moves_away_from_coin[is_my_coin == 0]
        return coop_moves


# DiCE operator
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

    def add_end_state_v(self, v):
        self.end_state_v = v


    def dice_objective(self):
        self_logprobs = torch.stack(self.self_logprobs, dim=1)
        other_logprobs = torch.stack(self.other_logprobs, dim=1)

        rewards = torch.stack(self.rewards, dim=1)

        # apply discount:
        cum_discount = torch.cumprod(args.gamma * torch.ones(*rewards.size()),
                                     dim=1).to(device) / args.gamma
        discounted_rewards = rewards * cum_discount

        if use_baseline:
            values = torch.stack(self.values, dim=1)
            # discounted_values = values * cum_discount

        # stochastics nodes involved in rewards dependencies:
        dependencies = torch.cumsum(self_logprobs + other_logprobs, dim=1)

        # logprob of all stochastic nodes:
        stochastic_nodes = self_logprobs + other_logprobs

        use_loaded_dice = False
        if use_baseline:
            use_loaded_dice = True

        if use_loaded_dice:
            next_val_history = torch.zeros(
                (args.batch_size, args.len_rollout),
                device=device)
            next_val_history[:, :args.len_rollout - 1] = values[:, 1:args.len_rollout]
            next_val_history[:, -1] = self.end_state_v

            if args.zero_vals:
                next_val_history = torch.zeros_like(next_val_history)
                values = torch.zeros_like(values)

            advantages = torch.zeros_like(values)
            lambd = args.gae_lambda # 1 here is essentially monte carlo (but with extrapolation of value in the end state)
            deltas = rewards + args.gamma * next_val_history.detach() - values.detach()
            gae = torch.zeros_like(deltas[:, 0]).float()
            for i in range(deltas.size(1) - 1, -1, -1):
                gae = gae * args.gamma * lambd + deltas[:, i]
                advantages[:, i] = gae

            discounts = torch.cumprod(
                args.gamma * torch.ones((args.len_rollout), device=device),
                dim=0) / args.gamma

            discounted_advantages = advantages * discounts

            deps_up_to_t = (torch.cumsum(stochastic_nodes, dim=1))

            deps_less_than_t = deps_up_to_t - stochastic_nodes  # take out the dependency in the given time step

            # Look at Loaded DiCE and GAE papers to see where this formulation comes from
            loaded_dice_rewards = ((magic_box(deps_up_to_t) - magic_box(
                deps_less_than_t)) * discounted_advantages).sum(dim=1).mean(dim=0)

            dice_objective = loaded_dice_rewards

        else:
            # dice objective:
            dice_objective = torch.mean(
                torch.sum(magic_box(dependencies) * discounted_rewards, dim=1))

        return -dice_objective  # want to minimize -objective

    def value_loss(self):
        values = torch.stack(self.values, dim=1)
        rewards = torch.stack(self.rewards, dim=1)
        final_state_vals = self.end_state_v.detach()
        return value_loss(values, rewards, final_state_vals)


def value_loss(values, rewards, final_state_vals):
    # Fixed original value update which I'm almost certain is wrong

    discounts = torch.cumprod(
        args.gamma * torch.ones((args.len_rollout), device=device),
        dim=0) / args.gamma

    gamma_t_r_ts = rewards * discounts
    G_ts = reverse_cumsum(gamma_t_r_ts, dim=1)
    R_ts = G_ts / discounts

    final_val_discounted_to_curr = (args.gamma * discounts.flip(dims=[0])).expand((final_state_vals.shape[0], discounts.shape[0])) \
                                   * final_state_vals.expand((discounts.shape[0], final_state_vals.shape[0])).t()

    # You DO need a detach on these. Because it's the target - it should be detached. It's a target value.
    # Essentially a Monte Carlo style type return for R_t, except for the final state we also use the estimated final state value.
    # This becomes our target for the value function loss. So it's kind of a mix of Monte Carlo and bootstrap, but anyway you need the final value
    # because otherwise your value calculations will be inconsistent
    values_loss = (R_ts + final_val_discounted_to_curr - values) ** 2
    values_loss = values_loss.sum(dim=1).mean(dim=0)

    print("Values loss")
    print(values_loss)
    return values_loss


# Pass stuff through GRU
def apply(batch_states, theta, hidden):
    batch_states = batch_states.flatten(start_dim=1)

    if args.hist_one:
        x = batch_states.matmul(theta[0])
        x = theta[1] + x
        x = torch.relu(x)
        x = x.matmul(theta[2])
        x = theta[3] + x
        out = x
        hy = None

    else:
        x = batch_states.matmul(theta[0])
        x = theta[1] + x

        x = torch.relu(x)

        gate_x = x.matmul(theta[2])
        gate_x = gate_x + theta[3]

        gate_h = hidden.matmul(theta[4])
        gate_h = gate_h + theta[5]

        i_r, i_i, i_n = gate_x.chunk(3, 1)
        h_r, h_i, h_n = gate_h.chunk(3, 1)

        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + (resetgate * h_n))

        hy = newgate + inputgate * (hidden - newgate)

        out = hy.matmul(theta[6])
        out = out + theta[7]

    return hy, out


def act(batch_states, theta_p, theta_v, h_p, h_v, ret_logits=False):
    h_p, logits = apply(batch_states, theta_p, h_p)
    categorical_act_probs = torch.softmax(logits, dim=-1)
    if use_baseline:
        h_v, values = apply(batch_states, theta_v, h_v)
        ret_vals = values.squeeze(-1)
    else:
        h_v, values = None, None
        ret_vals = None
    dist = Categorical(categorical_act_probs)
    actions = dist.sample()
    log_probs_actions = dist.log_prob(actions)

    if ret_logits:
        return actions, log_probs_actions, ret_vals, h_p, h_v, categorical_act_probs, logits
    return actions, log_probs_actions, ret_vals, h_p, h_v, categorical_act_probs


def get_gradient(objective, theta):
    # create differentiable gradient for 2nd orders:
    grad_objective = torch.autograd.grad(objective, (theta), create_graph=True)
    return grad_objective


def eval_vs_fixed_strategy(theta, values, strat="alld", i_am_red_agent=True):
    # just to evaluate progress:
    (s1, s2) = env.reset()
    score1 = 0
    score2 = 0
    h_p, h_v = (
        torch.zeros(args.batch_size, args.hidden_size).to(device),
        torch.zeros(args.batch_size, args.hidden_size).to(device))


    for t in range(args.len_rollout):
        if t > 0:
            prev_a = a

        if i_am_red_agent:
            s = s1
        else:
            s = s2

        a, lp, v1, h_p, h_v, cat_act_probs = act(s, theta, values, h_p,
                                                      h_v)
        if strat == "alld":
            if args.env == "ipd":
                # Always defect
                a_opp = torch.zeros_like(a)
            else:
                # Coin game
                # if I am red agent, I want to evaluate the other agent from the blue agent perspective
                a_opp = env.get_moves_shortest_path_to_coin(red_agent_perspective=(not i_am_red_agent))

        elif strat == "allc":
            if args.env == "ipd":
                # Always cooperate
                a_opp = torch.ones_like(a)
            else:
                a_opp = env.get_coop_action(red_agent_perspective=(not i_am_red_agent))
        elif strat == "tft":
            if args.env == "ipd":
                if t == 0:
                    # start with coop
                    a_opp = torch.ones_like(a)
                else:
                    # otherwise copy the last move of the other agent
                    a_opp = prev_a
            else:
                if t == 0:
                    a_opp = env.get_coop_action(
                        red_agent_perspective=(not i_am_red_agent))
                    prev_agent_coin_collected_same_col = torch.ones_like(a) # 0 = defect, collect other agent coin
                else:
                    if i_am_red_agent:
                        r_opp = r2
                    else:
                        r_opp = r1
                    # Agent here means me, the agent we are testing
                    prev_agent_coin_collected_same_col[r_opp < 0] = 0 # opp got negative reward from other agent collecting opp's coin
                    prev_agent_coin_collected_same_col[r_opp > 0] = 1 # opp is allowed to get positive reward from collecting own coin

                    a_opp_defect = env.get_moves_shortest_path_to_coin(red_agent_perspective=(not i_am_red_agent))
                    a_opp_coop = env.get_coop_action(red_agent_perspective=(not i_am_red_agent))

                    a_opp = torch.clone(a_opp_coop.detach())
                    a_opp[prev_agent_coin_collected_same_col == 0] = a_opp_defect[prev_agent_coin_collected_same_col == 0]

        else:
            raise NotImplementedError

        if i_am_red_agent:
            a1 = a
            a2 = a_opp
        else:
            a1 = a_opp
            a2 = a

        (s1, s2), (r1, r2), _, info = env.step((a1, a2))

        score1 += torch.mean(r1) / float(args.len_rollout)
        score2 += torch.mean(r2) / float(args.len_rollout)

    return (score1, score2), None


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
    if args.env == "coin" or args.env == "ogcoin":
        rr_matches_record, rb_matches_record, br_matches_record, bb_matches_record = 0., 0., 0., 0.

    for t in range(args.len_rollout):
        a1, lp1, v1, h_p1, h_v1, cat_act_probs1 = act(s1, theta1, values1, h_p1, h_v1)
        a2, lp2, v2, h_p2, h_v2, cat_act_probs2 = act(s2, theta2, values2, h_p2, h_v2)
        (s1, s2), (r1, r2), _, info = env.step((a1, a2))
        # cumulate scores
        score1 += torch.mean(r1) / float(args.len_rollout)
        score2 += torch.mean(r2) / float(args.len_rollout)

        if args.env == "coin" or args.env == "ogcoin":
            rr_matches, rb_matches, br_matches, bb_matches = info
            rr_matches_record += rr_matches
            rb_matches_record += rb_matches
            br_matches_record += br_matches
            bb_matches_record += bb_matches

    if args.env == "coin" or args.env == "ogcoin":
        return (score1, score2), (rr_matches_record, rb_matches_record, br_matches_record, bb_matches_record)

    return (score1, score2), None

class Agent():
    def __init__(self, input_size, hidden_size, action_size, lr_p, lr_v, theta_p=None, theta_v=None):
        self.hidden_size = hidden_size
        if args.hist_one:
            self.theta_p = nn.ParameterList([
                # Linear 1
                nn.Parameter(
                    torch.zeros((input_size, hidden_size), requires_grad=True)),
                nn.Parameter(torch.zeros(hidden_size, requires_grad=True)),

                # Linear 2
                nn.Parameter(
                    torch.zeros((hidden_size, action_size), requires_grad=True)),
                nn.Parameter(torch.zeros(action_size, requires_grad=True)),
            ]).to(device)

            self.theta_v = nn.ParameterList([
                # Linear 1
                nn.Parameter(
                    torch.zeros((input_size, hidden_size), requires_grad=True)),
                nn.Parameter(torch.zeros(hidden_size, requires_grad=True)),

                # Linear 2
                nn.Parameter(
                    torch.zeros((hidden_size, 1),
                                requires_grad=True)),
                nn.Parameter(torch.zeros(1, requires_grad=True)),
            ]).to(device)
        else:
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

        if theta_p is not None:
            self.theta_p = theta_p
        if theta_v is not None:
            self.theta_v = theta_v

        if args.optim.lower() == 'adam':
            self.theta_optimizer = torch.optim.Adam(self.theta_p, lr=lr_p)
            self.value_optimizer = torch.optim.Adam(self.theta_v, lr=lr_v)
        elif args.optim.lower() == 'sgd':
            self.theta_optimizer = torch.optim.SGD(self.theta_p, lr=lr_p)
            self.value_optimizer = torch.optim.SGD(self.theta_v, lr=lr_v)
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
        # Perhaps really should not be named 1
        # Well this also doesn't even have to be other, this works fine for any theta and vals as long as the state history is correct (corresponds to the theta and values you are using)
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

    def get_other_logits_values_for_states(self, other_theta, other_values, state_history):
        # Same comments as above. Questionable variable naming here
        h_p1, h_v1 = (
            torch.zeros(args.batch_size, self.hidden_size).to(device),
            torch.zeros(args.batch_size, self.hidden_size).to(device))

        logits_hist = []
        vals_hist = []


        for t in range(args.len_rollout):
            s1 = state_history[t]
            a1, lp1, v1, h_p1, h_v1, cat_act_probs1, logits = act(s1, other_theta,
                                                          other_values, h_p1,
                                                          h_v1, ret_logits=True)
            logits_hist.append(logits)
            vals_hist.append(v1)

        final_state = state_history[-1]
        # act just to get the final state values
        a1, lp1, v1, h_p1, h_v1, cat_act_probs1 = act(final_state, other_theta,
                                                      other_values,
                                                      h_p1, h_v1)
        final_state_vals = v1

        if use_baseline:
            return torch.stack(logits_hist, dim=1), torch.stack(vals_hist, dim=1), final_state_vals
        else:
            return torch.stack(logits_hist, dim=1), None, None


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

        # act just to get the final state values
        a2, lp2, v2, h_p2, h_v2, cat_act_probs2 = act(s2, other_theta,
                                                      other_values,
                                                      h_p2, h_v2)
        other_memory.add_end_state_v(v2)

        if not first_inner_step:
            curr_pol_probs = self.get_other_policies_for_states(other_theta, other_values, self.other_state_history)
            kl_div = torch.nn.functional.kl_div(torch.log(curr_pol_probs), self.ref_cat_act_probs_other.detach(), log_target=False, reduction='batchmean')
            print(kl_div)

        other_objective = other_memory.dice_objective()
        if not first_inner_step:
            other_objective += args.inner_beta * kl_div # we want to min kl div

        grad = get_gradient(other_objective, other_theta)

        if first_inner_step:
            # use as ref for KL div calc
            self.ref_cat_act_probs_other = torch.stack(cat_act_probs_other, dim=1)
            self.other_state_history = torch.stack(other_state_history, dim=0)

        return grad

    def out_lookahead(self, other_theta, other_values, first_outer_step=False, agent_copy_for_val_update=None):
        # AGENT COPY IS A COPY OF SELF, NOT OF OTHER. Used so that you can update the value function
        # while POLA is running in the outer loop, but the outer loop steps still calculate the loss
        # for the policy based on the old, static value function (this is for the val_update_after_loop stuff).
        # This should hopefully help with issues that might arise if value function is changing during the POLA update
        (s1, s2) = env.reset()
        memory = Memory()
        h_p1, h_v1, h_p2, h_v2 = (
        torch.zeros(args.batch_size, self.hidden_size).to(device),
        torch.zeros(args.batch_size, self.hidden_size).to(device),
        torch.zeros(args.batch_size, self.hidden_size).to(device),
        torch.zeros(args.batch_size, self.hidden_size).to(device))

        state_history_for_vals = []
        state_history_for_vals.append(s1)
        rew_history_for_vals = []

        if first_outer_step:
            cat_act_probs_self = []
            state_history_for_kl_div = []
            state_history_for_kl_div.append(s1)
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
                state_history_for_kl_div.append(s1)
            if args.ent_reg > 0:
                ent_vals.append(cat_act_probs1)
            state_history_for_vals.append(s1)
            rew_history_for_vals.append(r1)

        # act just to get the final state values
        a1, lp1, v1, h_p1, h_v1, cat_act_probs1 = act(s1, self.theta_p,
                                                      self.theta_v,
                                                      h_p1, h_v1)
        memory.add_end_state_v(v1)

        if not first_outer_step:
            curr_pol_probs = self.get_policies_for_states()
            kl_div = torch.nn.functional.kl_div(torch.log(curr_pol_probs), self.ref_cat_act_probs.detach(), log_target=False, reduction='batchmean')
            print(kl_div)

        # update self theta
        objective = memory.dice_objective()
        if not first_outer_step:
            objective += args.outer_beta * kl_div # we want to min kl div
        if args.ent_reg > 0:
            ent_vals = torch.stack(ent_vals, dim=0)
            ent_calc = - (ent_vals * torch.log(ent_vals)).sum(dim=-1).mean()
            objective += -ent_calc * args.ent_reg # but we want to max entropy (min negative entropy)
        self.theta_update(objective)
        # update self value:
        if use_baseline and not args.val_update_after_loop:
            v_loss = memory.value_loss()
            self.value_update(v_loss)

        if first_outer_step:
            self.ref_cat_act_probs = torch.stack(cat_act_probs_self, dim=1)
            self.state_history = torch.stack(state_history_for_kl_div, dim=0)

        if args.val_update_after_loop:
            assert agent_copy_for_val_update is not None
            state_history_for_vals = torch.stack(state_history_for_vals, dim=0)
            rew_history_for_vals = torch.stack(rew_history_for_vals, dim=1)
            curr_pol_logits, curr_vals, final_state_vals = self.get_other_logits_values_for_states(
                agent_copy_for_val_update.theta_p, agent_copy_for_val_update.theta_v, state_history_for_vals)
            v_loss = value_loss(values=curr_vals, rewards=rew_history_for_vals,
                                final_state_vals=final_state_vals)
            agent_copy_for_val_update.value_update(v_loss)


    def rollout_collect_data_for_opp_model(self, other_theta_p, other_theta_v):
        (s1, s2) = env.reset()
        memory = Memory()
        h_p1, h_v1, h_p2, h_v2 = (
            torch.zeros(args.batch_size, self.hidden_size).to(device),
            torch.zeros(args.batch_size, self.hidden_size).to(device),
            torch.zeros(args.batch_size, self.hidden_size).to(device),
            torch.zeros(args.batch_size, self.hidden_size).to(device))

        state_history, other_state_history = [], []
        state_history.append(s1)
        other_state_history.append(s2)
        act_history, other_act_history = [], []
        other_rew_history = []


        for t in range(args.len_rollout):
            a1, lp1, v1, h_p1, h_v1, cat_act_probs1 = act(s1, self.theta_p,
                                                          self.theta_v, h_p1,
                                                          h_v1)
            a2, lp2, v2, h_p2, h_v2, cat_act_probs2 = act(s2, other_theta_p,
                                                          other_theta_v, h_p2,
                                                          h_v2)
            (s1, s2), (r1, r2), _, _ = env.step((a1, a2))
            memory.add(lp1, lp2, v1, r1)

            state_history.append(s1)
            other_state_history.append(s2)
            act_history.append(a1)
            other_act_history.append(a2)
            other_rew_history.append(r2)

        # Stacking dim = 0 gives (len_rollout, batch)
        # Stacking dim = 1 gives (batch, len_rollout)
        state_history = torch.stack(state_history, dim=0)
        other_state_history = torch.stack(other_state_history, dim=0)
        act_history = torch.stack(act_history, dim=1)
        other_act_history = torch.stack(other_act_history, dim=1)

        other_rew_history = torch.stack(other_rew_history, dim=1)
        return state_history, other_state_history, act_history, other_act_history, other_rew_history


    def opp_model(self, om_lr_p, om_lr_v, true_other_theta_p, true_other_theta_v,
                  prev_model_theta_p=None, prev_model_theta_v=None):
        # true_other_theta_p and true_other_theta_v used only in the collection of data (rollouts in the environment)
        # so then this is not cheating. We do not assume access to other agent policy parameters (at least not direct, white box access)
        # We assume ability to collect trajectories through rollouts/play with the other agent in the environment
        # Essentially when using OM, we are now no longer doing dice update on the trajectories collected directly (which requires parameter access)
        # instead we collect the trajectories first, then build an OM, then rollout using OM and make DiCE/LOLA/POLA update based on that OM
        # Instead of direct rollout using opponent true parameters and update based on that.
        agent_opp = Agent(input_size, args.hidden_size, action_size, om_lr_p, om_lr_v, prev_model_theta_p, prev_model_theta_v)

        opp_model_data_batches = args.opp_model_data_batches

        for batch in range(opp_model_data_batches):
            # should in principle only do 1 collect, but I can do multiple "batches"
            # where repeating the below would be the same as collecting one big batch of environment interaction
            state_history, other_state_history, act_history, other_act_history, other_rew_history =\
                self.rollout_collect_data_for_opp_model(true_other_theta_p, true_other_theta_v)

            opp_model_iters = 0
            opp_model_steps_per_data_batch = args.opp_model_steps_per_batch

            other_act_history = torch.nn.functional.one_hot(other_act_history,
                                                            action_size)

            print(f"Opp Model Data Batch: {batch + 1}")

            for opp_model_iter in range(opp_model_steps_per_data_batch):
                # POLICY UPDATE
                curr_pol_logits, curr_vals, final_state_vals = self.get_other_logits_values_for_states(agent_opp.theta_p,
                                                                   agent_opp.theta_v,
                                                                   other_state_history)


                # KL div: p log p - p log q
                # use p for target, since it has 0 and 1
                # Then p log p has no deriv so can drop it, with respect to model
                # then -p log q

                # Calculate targets based on the action history (other act history)
                # Essentially treat the one hot vector of actions as a class label, and then run supervised learning

                c_e_loss = - (other_act_history * torch.log_softmax(curr_pol_logits, dim=-1)).sum(dim=-1).mean()

                print(c_e_loss.item())

                agent_opp.theta_update(c_e_loss)

                if use_baseline:
                    # VALUE UPDATE
                    v_loss = value_loss(values=curr_vals, rewards=other_rew_history, final_state_vals=final_state_vals)
                    agent_opp.value_update(v_loss)

                opp_model_iters += 1

        return agent_opp.theta_p, agent_opp.theta_v


def play(agent1, agent2, n_lookaheads, outer_steps, use_opp_model=False): #,prev_scores=None, prev_coins_collected_info=None):
    joint_scores = []
    score_record = []
    # You could do something like the below and then modify the code to just be one continuous record that includes past values when loading from checkpoint
    # if prev_scores is not None:
    #     score_record = prev_scores
    # I'm tired though.
    vs_fixed_strats_score_record = [[], []]

    print("start iterations with", n_lookaheads, "inner steps and", outer_steps, "outer steps:")
    same_colour_coins_record = []
    diff_colour_coins_record = []
    coins_collected_info = (same_colour_coins_record, diff_colour_coins_record)

    agent2_theta_p_model, agent1_theta_p_model = None, None
    agent2_theta_v_model, agent1_theta_v_model = None, None

    for update in range(args.n_update):

        start_theta1 = [tp.detach().clone().requires_grad_(True) for tp in
                            agent1.theta_p]
        start_val1 = [tv.detach().clone().requires_grad_(True) for tv in
                        agent1.theta_v]
        start_theta2 = [tp.detach().clone().requires_grad_(True) for tp in
                            agent2.theta_p]
        start_val2 = [tv.detach().clone().requires_grad_(True) for tv in
                        agent2.theta_v]


        if use_opp_model:
            agent2_theta_p_model, agent2_theta_v_model = agent1.opp_model(args.om_lr_p, args.om_lr_v,
                                                    true_other_theta_p=start_theta2,
                                                    true_other_theta_v=start_val2,
                                                    prev_model_theta_p=agent2_theta_p_model)
            agent1_theta_p_model, agent1_theta_v_model = agent2.opp_model(args.om_lr_p, args.om_lr_v,
                                                    true_other_theta_p=start_theta1,
                                                    true_other_theta_v=start_val1,
                                                    prev_model_theta_p=agent1_theta_p_model)

        agent1_copy_for_val_update = None
        agent2_copy_for_val_update = None
        if args.val_update_after_loop:
            theta_p_1_copy_for_vals = [tp.detach().clone().requires_grad_(True) for tp in
                            agent1.theta_p]
            theta_v_1_copy_for_vals = [tv.detach().clone().requires_grad_(True) for tv in
                          agent1.theta_v]
            theta_p_2_copy_for_vals = [tp.detach().clone().requires_grad_(True) for tp in
                            agent2.theta_p]
            theta_v_2_copy_for_vals = [tv.detach().clone().requires_grad_(True) for tv in
                          agent2.theta_v]
            agent1_copy_for_val_update = Agent(input_size, args.hidden_size,
                                              action_size, args.lr_out,
                                              args.lr_v, theta_p_1_copy_for_vals,
                                              theta_v_1_copy_for_vals)
            agent2_copy_for_val_update = Agent(input_size, args.hidden_size,
                                              action_size, args.lr_out,
                                              args.lr_v, theta_p_2_copy_for_vals,
                                              theta_v_2_copy_for_vals)

        for outer_step in range(outer_steps):
            # copy other's parameters:
            th2_to_copy = start_theta2
            val2_to_copy = start_val2
            if use_opp_model:
                th2_to_copy = agent2_theta_p_model
                val2_to_copy = agent2_theta_v_model

            theta2_ = [tp.detach().clone().requires_grad_(True) for tp in
                       th2_to_copy]
            values2_ = [tv.detach().clone().requires_grad_(True) for tv in
                        val2_to_copy]

            for inner_step in range(n_lookaheads):
                if inner_step == 0:
                    # estimate other's gradients from in_lookahead:
                    grad2 = agent1.in_lookahead(theta2_, values2_, first_inner_step=True)
                else:
                    grad2 = agent1.in_lookahead(theta2_, values2_, first_inner_step=False)
                # update other's theta: NOTE HERE THIS IS JUST AN SGD UPDATE
                theta2_ = [theta2_[i] - args.lr_in * grad2[i] for i in
                           range(len(theta2_))]



            # update own parameters from out_lookahead:
            if outer_step == 0:
                agent1.out_lookahead(theta2_, values2_, first_outer_step=True, agent_copy_for_val_update=agent1_copy_for_val_update)
            else:
                agent1.out_lookahead(theta2_, values2_, first_outer_step=False, agent_copy_for_val_update=agent1_copy_for_val_update)

            if args.print_info_each_outer_step:
                print("Agent 1 Sample Obs Info:")
                print_policy_and_value_info(agent1.theta_p, agent1.theta_v)
                if use_opp_model:
                    print("Agent 1 Updated Opp Model of Agent 2:")
                    print_policy_and_value_info(theta2_, values2_)

        for outer_step in range(outer_steps):
            th1_to_copy = start_theta1
            val1_to_copy = start_val1
            if use_opp_model:
                th1_to_copy = agent1_theta_p_model
                val1_to_copy = agent1_theta_v_model

            theta1_ = [tp.detach().clone().requires_grad_(True) for tp in
                       th1_to_copy]
            values1_ = [tv.detach().clone().requires_grad_(True) for tv in
                        val1_to_copy]

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
                agent2.out_lookahead(theta1_, values1_, first_outer_step=True, agent_copy_for_val_update=agent2_copy_for_val_update)
            else:
                agent2.out_lookahead(theta1_, values1_, first_outer_step=False, agent_copy_for_val_update=agent2_copy_for_val_update)

            if args.print_info_each_outer_step:
                print("Agent 2 Sample Obs Info:")
                print_policy_and_value_info(agent2.theta_p, agent2.theta_v)
                if use_opp_model:
                    print("Agent 2 Updated Opp Model of Agent 1:")
                    print_policy_and_value_info(theta1_,
                                                values1_)

        if args.val_update_after_loop:
            updated_theta_v_1 = [tv.detach().clone().requires_grad_(True)
                                       for tv in
                                       agent1_copy_for_val_update.theta_v]
            updated_theta_v_2 = [tv.detach().clone().requires_grad_(True)
                                       for tv in
                                       agent2_copy_for_val_update.theta_v]
            agent1.theta_v = updated_theta_v_1
            agent2.theta_v = updated_theta_v_2

        # evaluate progress:
        score, info = step(agent1.theta_p, agent2.theta_p, agent1.theta_v,
                           agent2.theta_v)

        print("Eval vs Fixed Strategies:")
        score1rec = []
        score2rec = []
        for strat in ["alld", "allc", "tft"]:
            print(f"Playing against strategy: {strat.upper()}")
            score1, _ = eval_vs_fixed_strategy(agent1.theta_p, agent1.theta_v, strat, i_am_red_agent=True)
            score1rec.append(score1[0])
            print(f"Agent 1 score: {score1[0]}")
            score2, _ = eval_vs_fixed_strategy(agent2.theta_p, agent2.theta_v, strat, i_am_red_agent=False)
            score2rec.append(score2[1])
            print(f"Agent 2 score: {score2[1]}")

            print(score1)
            print(score2)

        score1rec = torch.stack(score1rec)
        score2rec = torch.stack(score2rec)
        vs_fixed_strats_score_record[0].append(score1rec)
        vs_fixed_strats_score_record[1].append(score2rec)

        if args.env == "coin" or args.env == "ogcoin":
            rr_matches, rb_matches, br_matches, bb_matches = info
            same_colour_coins = (rr_matches + bb_matches).item()
            diff_colour_coins = (rb_matches + br_matches).item()
            same_colour_coins_record.append(same_colour_coins)
            diff_colour_coins_record.append(diff_colour_coins)
        joint_scores.append(0.5 * (score[0] + score[1]))
        score = torch.stack(score)
        score_record.append(score)

        # print
        if update % args.print_every == 0:
            print("*" * 10)
            print("Epoch: {}".format(update + 1), flush=True)
            print(f"Score 0: {score[0]}")
            print(f"Score 1: {score[1]}")
            if args.env == "coin" or args.env == "ogcoin":
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
            if use_opp_model:
                print("Agent 1 Opp Model of Agent 2:")
                print_policy_and_value_info(agent2_theta_p_model,
                                            agent2_theta_v_model)
                print("Agent 2 Opp Model of Agent 1:")
                print_policy_and_value_info(agent1_theta_p_model,
                                            agent1_theta_v_model)

        if (update + 1) % args.checkpoint_every == 0:
            now = datetime.datetime.now()
            checkpoint(agent1, agent2, coins_collected_info, score_record, vs_fixed_strats_score_record,
                       "checkpoint_{}_{}_seed{}.pt".format(update + 1, now.strftime(
                           '%Y-%m-%d_%H-%M'), args.seed), args)

    return joint_scores




if __name__ == "__main__":
    parser = argparse.ArgumentParser("NPLOLA")
    parser.add_argument("--inner_steps", type=int, default=1, help="inner loop steps for DiCE")
    parser.add_argument("--outer_steps", type=int, default=1, help="outer loop steps for POLA")
    parser.add_argument("--lr_out", type=float, default=0.005,
                        help="outer loop learning rate: same learning rate across all policies for now")
    parser.add_argument("--lr_in", type=float, default=0.03,
                        help="inner loop learning rate (eta): this has no use in the naive learning case. Used for the gradient step done for the lookahead for other agents during LOLA (therefore, often scaled to be higher than the outer learning rate in non-proximal LOLA). Note that this has a different meaning for the Taylor approx vs. actual update versions. A value of eta=1 is perfectly reasonable for the Taylor approx version as this balances the scale of the gradient with the naive learning term (and will be multiplied by the outer learning rate after), whereas for the actual update version with neural net, 1 is way too big an inner learning rate. For prox, this is the learning rate on the inner prox loop so is not that important - you want big enough to be fast-ish, but small enough to converge.")
    parser.add_argument("--lr_v", type=float, default=0.001,
                        help="same learning rate across all policies for now. Should be around maybe 0.001 or less for neural nets to avoid instability")
    parser.add_argument("--gamma", type=float, default=0.96, help="discount rate")
    parser.add_argument("--n_update", type=int, default=5000, help="number of epochs to run")
    parser.add_argument("--len_rollout", type=int, default=50, help="How long we want the time horizon of the game to be (number of steps before termination/number of iterations of the IPD)")
    parser.add_argument("--batch_size", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=1, help="for seed")
    parser.add_argument("--hidden_size", type=int, default=32)
    parser.add_argument("--print_every", type=int, default=1, help="Print every x number of epochs")
    parser.add_argument("--outer_beta", type=float, default=0.0, help="for outer kl penalty with POLA")
    parser.add_argument("--inner_beta", type=float, default=0.0, help="for inner kl penalty with POLA")
    parser.add_argument("--save_dir", type=str, default='.', help="Where to save checkpoints")
    parser.add_argument("--checkpoint_every", type=int, default=1000, help="Epochs between checkpoint save")
    parser.add_argument("--load_path", type=str, default=None, help="Give path if loading from a checkpoint")
    parser.add_argument("--ent_reg", type=float, default=0.0, help="entropy regularizer")
    parser.add_argument("--diff_coin_reward", type=float, default=1.0, help="changes problem setting (the reward for picking up coin of different colour)")
    parser.add_argument("--diff_coin_cost", type=float, default=-2.0, help="changes problem setting (the cost to the opponent when you pick up a coin of their colour)")
    parser.add_argument("--same_coin_reward", type=float, default=1.0, help="changes problem setting (the reward for picking up coin of same colour)")
    parser.add_argument("--grid_size", type=int, default=3, help="Grid size for Coin Game")
    parser.add_argument("--optim", type=str, default="adam", help="Used only for the outer agent (in the out_lookahead)")
    parser.add_argument("--no_baseline", action="store_true", help="Use NO Baseline (critic) for variance reduction. Default is baseline using Loaded DiCE with GAE")
    parser.add_argument("--opp_model", action="store_true", help="Use Opponent Modeling")
    parser.add_argument("--opp_model_steps_per_batch", type=int, default=1, help="How many steps to train opp model on each batch at the beginning of each POLA epoch")
    parser.add_argument("--opp_model_data_batches", type=int, default=100, help="How many batches of data (right now from rollouts) to train opp model on")
    parser.add_argument("--om_lr_p", type=float, default=0.005,
                        help="learning rate for opponent modeling (imitation/supervised learning) for policy")
    parser.add_argument("--om_lr_v", type=float, default=0.001,
                        help="learning rate for opponent modeling (imitation/supervised learning) for value")
    parser.add_argument("--env", type=str, default="ogcoin",
                        choices=["ipd", "ogcoin"])
    parser.add_argument("--hist_one", action="store_true", help="Use one step history (no gru or rnn, just one step history)")
    parser.add_argument("--print_info_each_outer_step", action="store_true", help="For debugging/curiosity sake")
    parser.add_argument("--init_state_coop", action="store_true", help="For IPD only: have the first state be CC instead of a separate start state")
    parser.add_argument("--split_coins", action="store_true", help="If true, then when both agents step on same coin, each gets 50% of the reward as if they were the only agent collecting that coin. Only tested with OGCoin so far")
    parser.add_argument("--zero_vals", action="store_true", help="For testing/debug. Can also serve as another way to do no_baseline. Set all values to be 0 in Loaded Dice Calculation")
    parser.add_argument("--gae_lambda", type=float, default=1,
                        help="lambda for GAE (1 = monte carlo style, 0 = TD style)")
    parser.add_argument("--val_update_after_loop", action="store_true", help="Update values only after outer POLA loop finishes, not during the POLA loop")

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.env == "ogcoin":
        input_size = args.grid_size ** 2 * 4
        action_size = 4
        env = OGCoinGameGPU(max_steps=args.len_rollout, batch_size=args.batch_size, grid_size=args.grid_size)
    elif args.env == "ipd":
        # input_size = 2 * 2 # n agents by n agents
        input_size = 3 * 2 # one hot repr dim by n agents
        action_size = 2
        env = IteratedPrisonersDilemma(max_steps=args.len_rollout, batch_size=args.batch_size)
    else:
        raise NotImplementedError("Unknown Environment")

    if args.load_path is None:
        agent1 = Agent(input_size, args.hidden_size, action_size, lr_p=args.lr_out, lr_v = args.lr_v)
        agent2 = Agent(input_size, args.hidden_size, action_size, lr_p=args.lr_out, lr_v = args.lr_v)
    else:
        agent1, agent2, coins_collected_info, prev_scores, vs_fixed_strat_scores = load_from_checkpoint()
        print(torch.stack(prev_scores))

    use_baseline = True
    if args.no_baseline:
        use_baseline = False

    joint_scores = play(agent1, agent2, args.inner_steps, args.outer_steps, args.opp_model)
