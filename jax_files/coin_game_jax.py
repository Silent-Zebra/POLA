import jax
import jax.numpy as jnp
from typing import NamedTuple
from typing import Tuple


class CoinGameState(NamedTuple):
    red_pos: jnp.ndarray
    blue_pos: jnp.ndarray
    coin_pos: jnp.ndarray
    is_red_coin: jnp.ndarray
    step_count: jnp.ndarray


# class CoinGameJAX:
MOVES = jax.device_put(
    jnp.array(
        [
            [0, 1], # right
            [0, -1], # left
            [1, 0], # down
            [-1, 0], # up
        ]
    )
)


class CoinGame:
    def __init__(self, grid_size=3):
        self.grid_size = grid_size
        pass

    def generate_coins(self, subkey, red_pos_flat, blue_pos_flat):
        # if red_pos_flat == blue_pos_flat:
        #     coin_pos_max_val = self.grid_size ** 2 - 1
        # else:
        #     coin_pos_max_val = self.grid_size ** 2 - 2

        subkey, sk1, sk2 = jax.random.split(subkey, 3)

        coin_pos_max_val = jax.random.randint(sk1, shape=(1,), minval=0, maxval=0) + + self.grid_size ** 2 - 2

        # print(red_pos_flat)
        # print(blue_pos_flat)

        coin_pos_max_val += (red_pos_flat == blue_pos_flat)

        stacked_pos = jnp.stack((red_pos_flat, blue_pos_flat))
        # print(stacked_pos)
        # print(stacked_pos.shape)


        min_pos = jnp.min(stacked_pos)
        max_pos = jnp.max(stacked_pos)

        # print(min_pos)
        # print(max_pos)
        # print(min_pos.shape)

        coin_pos_flat = jax.random.randint(sk2, shape=(1,), minval=0,
                                           maxval=coin_pos_max_val)

        coin_pos_flat += (coin_pos_flat >= min_pos)
        coin_pos_flat += jnp.logical_and((coin_pos_flat >= max_pos), (red_pos_flat != blue_pos_flat))



        # print(coin_pos_flat)

        coin_pos = jnp.stack((coin_pos_flat // self.grid_size, coin_pos_flat % self.grid_size)).squeeze(-1)

        return coin_pos

    def reset(self, subkey) -> Tuple[jnp.ndarray, CoinGameState]:
        subkey, sk1, sk2, sk3 = jax.random.split(subkey, 4)

        # 3x3 coin game
        red_pos_flat = jax.random.randint(sk1, shape=(1,), minval=0, maxval=self.grid_size ** 2)
        # print(red_pos_flat)
        # print(red_pos_flat // self.grid_size)
        # print(red_pos_flat % self.grid_size)
        red_pos = jnp.stack((red_pos_flat // self.grid_size, red_pos_flat % self.grid_size)).squeeze(-1)


        blue_pos_flat = jax.random.randint(sk2, shape=(1,), minval=0, maxval=self.grid_size ** 2)
        blue_pos = jnp.stack((blue_pos_flat // self.grid_size, blue_pos_flat % self.grid_size)).squeeze(-1)

        # if red_pos_flat == blue_pos_flat:
        #     coin_pos_max_val = self.grid_size ** 2 - 1
        # else:
        #     coin_pos_max_val = self.grid_size ** 2
        #
        # min_pos = jnp.min(red_pos_flat, blue_pos_flat)
        # max_pos = jnp.max(red_pos_flat, blue_pos_flat)
        #
        # coin_pos_flat = jax.random.randint(sk3, shape=(1,), minval=0, maxval=coin_pos_max_val)
        # if coin_pos_flat >= min_pos:
        #     coin_pos_flat += 1
        # if coin_pos_flat >= max_pos and (red_pos_flat != blue_pos_flat):
        #     coin_pos_flat += 1
        #
        # coin_pos = jnp.ndarray(
        #     [coin_pos_flat // self.grid_size, coin_pos_flat % self.grid_size])

        coin_pos = self.generate_coins(sk3, red_pos_flat, blue_pos_flat)

        # print(red_pos.shape)
        # print(coin_pos.shape)
        # 1 / 0



        step_count = jnp.zeros(1)
        is_red_coin = jax.random.randint(sk3, shape=(1,), minval=0, maxval=2)

        state = CoinGameState(red_pos, blue_pos, coin_pos, is_red_coin, step_count)
        obs = self.state_to_obs(state)
        return state, obs

    def state_to_obs(self, state: CoinGameState) -> jnp.ndarray:

        # print(state.red_pos)
        # print(state.coin_pos)
        # print(state.red_pos[0])
        # print(state.coin_pos[0])
        is_red_coin = state.is_red_coin[0]
        # print(is_red_coin)
        obs = jnp.zeros((4, 3, 3))
        obs = obs.at[0, state.red_pos[0], state.red_pos[1]].set(1.0)
        obs = obs.at[1, state.blue_pos[0], state.blue_pos[1]].set(1.0)
        # red coin pos
        obs = obs.at[2, state.coin_pos[0], state.coin_pos[1]].set(is_red_coin)
        # blue coin pos
        obs = obs.at[3, state.coin_pos[0], state.coin_pos[1]].set(1.0 - is_red_coin)
        obs = obs.reshape(36)
        return obs

    def step(self, state: CoinGameState, action_0: int, action_1: int, subkey: jnp.ndarray) -> Tuple[jnp.ndarray, list]:


        new_red_pos = (state.red_pos + MOVES[action_0]) % 3
        new_blue_pos = (state.blue_pos + MOVES[action_1]) % 3

        is_red_coin = state.is_red_coin[0]

        zero_rew = jnp.zeros(1)

        red_red_matches = jnp.all(new_red_pos == state.coin_pos, axis=-1) * is_red_coin
        red_blue_matches = jnp.all(new_red_pos == state.coin_pos, axis=-1) * (1 - is_red_coin)

        red_reward = jnp.where(red_red_matches, zero_rew + 1, zero_rew)
        red_reward += jnp.where(red_blue_matches, zero_rew + 1, zero_rew)

        blue_red_matches = jnp.all(new_blue_pos == state.coin_pos, axis=-1) * is_red_coin
        blue_blue_matches = jnp.all(new_blue_pos == state.coin_pos, axis=-1) * (1 - is_red_coin)

        blue_reward = jnp.where(blue_red_matches, zero_rew + 1, zero_rew)
        blue_reward += jnp.where(blue_blue_matches, zero_rew + 1, zero_rew)

        red_reward += jnp.where(blue_red_matches, zero_rew - 2, zero_rew)
        blue_reward += jnp.where(red_blue_matches, zero_rew - 2, zero_rew)

        need_new_coins = ((red_red_matches + red_blue_matches + blue_red_matches + blue_blue_matches) > 0)
        flipped_is_red_coin = 1 - state.is_red_coin
        new_is_red_coin = need_new_coins * flipped_is_red_coin + (1 - need_new_coins) * state.is_red_coin

        # print(need_new_coins)
        # print(new_is_red_coin)
        #
        # print(new_red_pos)
        # print(new_blue_pos)
        # print(state.red_pos)
        # print(state.blue_pos)

        new_red_pos_flat = new_red_pos[0] * self.grid_size + new_red_pos[1]
        new_blue_pos_flat = new_blue_pos[0] * self.grid_size + new_blue_pos[1]
        # print(new_blue_pos_flat)


        generated_coins = self.generate_coins(subkey, new_red_pos_flat,
                                           new_blue_pos_flat)

        # print(need_new_coins)
        # print(generated_coins)
        # print(state.coin_pos)

        new_coin_pos = need_new_coins * generated_coins + (1-need_new_coins) * state.coin_pos

        # print(new_coin_pos)

        # if (red_red_matches + red_blue_matches + blue_red_matches + blue_blue_matches).sum() > 0:
        #     print(red_red_matches)
        #     print(red_blue_matches)
        #     print(blue_red_matches)
        #     print(blue_blue_matches)
        #     print((red_red_matches + red_blue_matches + blue_red_matches + blue_blue_matches).sum())
        #     new_is_red_coin = 1.0 - state.is_red_coin
        #     if jnp.all(new_red_pos) == jnp.all(new_blue_pos):
        #         1/0 # check that this happens sometimes
        #         coin_pos_max_val = self.grid_size ** 2 - 1
        #     else:
        #         coin_pos_max_val = self.grid_size ** 2
        #
        #     new_red_pos_flat = new_red_pos[:,0] * self.grid_size + new_red_pos[:, 1]
        #     new_blue_pos_flat = new_blue_pos[:,0] * self.grid_size + new_blue_pos[:, 1]
        #
        #     new_coin_pos = self.generate_coins(subkey, new_red_pos_flat, new_blue_pos_flat)
        #
        # else:
        #     assert (red_red_matches + red_blue_matches + blue_red_matches + blue_blue_matches).sum() == 0
        #     new_is_red_coin = state.is_red_coin
        #     new_coin_pos = state.coin_pos

        step_count = state.step_count + 1

        new_state = CoinGameState(new_red_pos, new_blue_pos, new_coin_pos, new_is_red_coin, step_count)
        obs = self.state_to_obs(new_state)

        return new_state, obs, (red_reward, blue_reward), (red_red_matches, red_blue_matches, blue_red_matches, blue_blue_matches)


    def get_moves_shortest_path_to_coin(self, red_agent_perspective=True):
        # Ties broken arbitrarily, in this case, since I check the vertical distance later
        # priority is given to closing vertical distance (making up or down moves)
        # before horizontal moves
        if red_agent_perspective:
            agent_pos = self.red_pos
        else:
            agent_pos = self.blue_pos
        actions = jax.random.randint(jax.random.PRNGKey(0), shape=(1,), minval=0, maxval=0)

        # assumes red agent perspective
        horiz_dist_right = (self.coin_pos[1] - agent_pos[1]) % self.grid_size
        horiz_dist_left = (agent_pos[1] - self.coin_pos[1]) % self.grid_size

        vert_dist_down = (self.coin_pos[0] - agent_pos[0]) % self.grid_size
        vert_dist_up = (agent_pos[0] - self.coin_pos[0]) % self.grid_size
        actions.at[horiz_dist_right < horiz_dist_left].set(0)
        actions.at[horiz_dist_left < horiz_dist_right].set(1)
        actions.at[vert_dist_down < vert_dist_up].set(2)
        actions.at[vert_dist_up < vert_dist_down].set(3)
        # Assumes no coin spawns under agent
        assert jnp.logical_and(horiz_dist_right == horiz_dist_left, vert_dist_down == vert_dist_up).sum() == 0

        print(actions)
        1/0

        return actions

    def get_moves_away_from_coin(self, moves_towards_coin):
        opposite_moves = jnp.zeros_like(moves_towards_coin)
        opposite_moves.at[moves_towards_coin == 0].set(1)
        opposite_moves.at[moves_towards_coin == 1].set(0)
        opposite_moves.at[moves_towards_coin == 2].set(3)
        opposite_moves.at[moves_towards_coin == 3].set(2)
        return opposite_moves

    def get_coop_action(self, red_agent_perspective=True):
        # move toward coin if same colour, away if opposite colour
        # An agent that always does this is considered to 'always cooperate'
        moves_towards_coin = self.get_moves_shortest_path_to_coin(red_agent_perspective=red_agent_perspective)
        moves_away_from_coin = self.get_moves_away_from_coin(moves_towards_coin)
        coop_moves = jnp.zeros_like(moves_towards_coin) - 1
        if red_agent_perspective:
            is_my_coin = self.is_red_coin
        else:
            is_my_coin = 1 - self.is_red_coin

        coop_moves.at[is_my_coin == 1].set(moves_towards_coin[is_my_coin == 1])
        coop_moves.at[is_my_coin == 0].set(moves_away_from_coin[is_my_coin == 0])
        return coop_moves
