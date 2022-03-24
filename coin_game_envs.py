

import numpy as np
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

"""
OG COIN GAME.
COIN CANNOT SPAWN UNDER EITHER AGENT.

NOTE: THIS ALSO DOES NOT FLIP THE OBS, WOULD HAVE TO MAKE THE SAME FIX HERE.
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
        coin_pos_flat[torch.logical_and(minpos==maxpos, torch.randn_like(minpos) < 1.0 / (self.grid_size*self.grid_size - 1))] = minpos+1 # THIS IS TO FIX THE OFFSET BUG
        coin_pos_flat = coin_pos_flat % (self.grid_size * self.grid_size)        
        self.coin_pos = torch.stack((coin_pos_flat // self.grid_size, coin_pos_flat % self.grid_size), dim=-1)

        state = self._generate_state()
        observations = [state, state]
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

        coin_pos_flat = self.coin_pos[:,0] * self.grid_size + self.coin_pos[:,1]
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
        red_reward = torch.zeros_like(self.red_coin)
        red_reward[red_matches] = 1

        blue_matches = self._same_pos(self.blue_pos, self.coin_pos)
        blue_reward = torch.zeros_like(self.red_coin)
        blue_reward[blue_matches] = 1

        red_reward[torch.logical_and(blue_matches, self.red_coin)] -= 2
        blue_reward[torch.logical_and(red_matches, 1 - self.red_coin)] -= 2

        self._generate_coins()
        reward = [red_reward.float(), blue_reward.float()]
        state = self._generate_state()
        observations = [state, state]
        if self.step_count >= self.max_steps:
            done = torch.ones_like(self.red_coin)
        else:
            done = torch.zeros_like(self.red_coin)

        return observations, reward, done


"""
THIS ONE IS THE ONE WHRE THE COINS CANT SPAWN ON THE SMAE SQUARE.
"""
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
        blue_coin_pos_flat = torch.randint(self.grid_size * self.grid_size - 1,
                                           size=(self.batch_size,)).to(device)
        blue_coin_pos_flat[blue_coin_pos_flat >= red_coin_pos_flat] += 1 

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

        mask_blue = torch.logical_or(
            self._same_pos(self.blue_coin_pos, self.blue_pos),
            self._same_pos(self.blue_coin_pos, self.red_pos))

        mask_red_only = torch.logical_and(mask_red, ~mask_blue)
        mask_blue_only = torch.logical_and(~mask_red, mask_blue)
        mask_both = torch.logical_and(mask_red, mask_blue)


        """
        COMPUTE RED COIN POS FOR WHEN BOTH RESET
        """
        red_coin_pos_flat = torch.randint(self.grid_size * self.grid_size,
                                          size=(self.batch_size,)).to(device)[
            mask_both]
        self.red_coin_pos[mask_both] = torch.stack((
                                                  torch.div(red_coin_pos_flat, self.grid_size, rounding_mode='floor'),
                                                  red_coin_pos_flat % self.grid_size),
                                                  dim=-1)
        """
        COMPUTE RED COIN POS FOR WHEN ONLY RED RESET
        """
        blue_coin_pos_flat = self.blue_coin_pos[mask_red_only,0] * self.grid_size + self.blue_coin_pos[mask_red_only, 1] 
        red_coin_pos_flat = torch.randint(self.grid_size * self.grid_size - 1,
                                          size=(self.batch_size,)).to(device)[
            mask_red_only]
        red_coin_pos_flat[red_coin_pos_flat >= blue_coin_pos_flat] += 1
        self.red_coin_pos[mask_red_only] = torch.stack((
                                                  torch.div(red_coin_pos_flat, self.grid_size, rounding_mode='floor'),
                                                  red_coin_pos_flat % self.grid_size),
                                                  dim=-1)

        """
        COMPUTE BLUE POS
        """
        red_coin_pos_flat = self.red_coin_pos[mask_blue, 0] * self.grid_size + self.red_coin_pos[mask_blue, 1]
        blue_coin_pos_flat = torch.randint(self.grid_size * self.grid_size - 1,
                                           size=(self.batch_size,)).to(device)[
            mask_blue]
        blue_coin_pos_flat[blue_coin_pos_flat >= red_coin_pos_flat] += 1
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


