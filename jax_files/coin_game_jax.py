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

    def generate_coins(self, subkey, red_pos_flat, blue_pos_flat):

        subkey, sk1, sk2 = jax.random.split(subkey, 3)

        coin_pos_max_val = jax.random.randint(sk1, shape=(1,), minval=0, maxval=0) + + self.grid_size ** 2 - 2

        coin_pos_max_val += (red_pos_flat == blue_pos_flat)

        stacked_pos = jnp.stack((red_pos_flat, blue_pos_flat))

        min_pos = jnp.min(stacked_pos)
        max_pos = jnp.max(stacked_pos)

        coin_pos_flat = jax.random.randint(sk2, shape=(1,), minval=0,
                                           maxval=coin_pos_max_val)

        coin_pos_flat += (coin_pos_flat >= min_pos)
        coin_pos_flat += jnp.logical_and((coin_pos_flat >= max_pos), (red_pos_flat != blue_pos_flat))


        coin_pos = jnp.stack((coin_pos_flat // self.grid_size, coin_pos_flat % self.grid_size)).squeeze(-1)

        return coin_pos

    def reset(self, subkey) -> Tuple[jnp.ndarray, CoinGameState]:
        subkey, sk1, sk2, sk3 = jax.random.split(subkey, 4)

        red_pos_flat = jax.random.randint(sk1, shape=(1,), minval=0, maxval=self.grid_size ** 2)

        red_pos = jnp.stack((red_pos_flat // self.grid_size, red_pos_flat % self.grid_size)).squeeze(-1)


        blue_pos_flat = jax.random.randint(sk2, shape=(1,), minval=0, maxval=self.grid_size ** 2)
        blue_pos = jnp.stack((blue_pos_flat // self.grid_size, blue_pos_flat % self.grid_size)).squeeze(-1)

        coin_pos = self.generate_coins(sk3, red_pos_flat, blue_pos_flat)

        step_count = jnp.zeros(1)
        is_red_coin = jax.random.randint(sk3, shape=(1,), minval=0, maxval=2)

        state = CoinGameState(red_pos, blue_pos, coin_pos, is_red_coin, step_count)
        obs = self.state_to_obs(state)
        return state, obs

    def state_to_obs(self, state: CoinGameState) -> jnp.ndarray:

        is_red_coin = state.is_red_coin[0]
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

        new_red_pos_flat = new_red_pos[0] * self.grid_size + new_red_pos[1]
        new_blue_pos_flat = new_blue_pos[0] * self.grid_size + new_blue_pos[1]


        generated_coins = self.generate_coins(subkey, new_red_pos_flat,
                                           new_blue_pos_flat)

        new_coin_pos = need_new_coins * generated_coins + (1-need_new_coins) * state.coin_pos

        step_count = state.step_count + 1

        new_state = CoinGameState(new_red_pos, new_blue_pos, new_coin_pos, new_is_red_coin, step_count)
        obs = self.state_to_obs(new_state)

        red_reward = red_reward.squeeze(-1)
        blue_reward = blue_reward.squeeze(-1)

        return new_state, obs, (red_reward, blue_reward), (red_red_matches, red_blue_matches, blue_red_matches, blue_blue_matches)

    def get_moves_shortest_path_to_coin(self, state, red_agent_perspective=True):
        # Ties broken arbitrarily, in this case, since I check the vertical distance later
        # priority is given to closing vertical distance (making up or down moves)
        # before horizontal moves
        if red_agent_perspective:
            agent_pos = state.red_pos
        else:
            agent_pos = state.blue_pos
        actions = jax.random.randint(jax.random.PRNGKey(0), shape=(1,), minval=0, maxval=0)

        # assumes red agent perspective
        horiz_dist_right = (state.coin_pos[:,1] - agent_pos[:,1]) % self.grid_size
        horiz_dist_left = (agent_pos[:,1] - state.coin_pos[:,1]) % self.grid_size

        vert_dist_down = (state.coin_pos[:,0] - agent_pos[:,0]) % self.grid_size
        vert_dist_up = (agent_pos[:,0] - state.coin_pos[:,0]) % self.grid_size
        actions = jnp.where(horiz_dist_right < horiz_dist_left, 0, actions)
        actions = jnp.where(horiz_dist_left < horiz_dist_right, 1, actions)
        actions = jnp.where(vert_dist_down < vert_dist_up, 2, actions)
        actions = jnp.where(vert_dist_up < vert_dist_down, 3, actions)

        return actions

    def get_moves_away_from_coin(self, moves_towards_coin):
        opposite_moves = jnp.zeros_like(moves_towards_coin)
        opposite_moves = jnp.where(moves_towards_coin == 0, 1, opposite_moves)
        opposite_moves = jnp.where(moves_towards_coin == 1, 0, opposite_moves)
        opposite_moves = jnp.where(moves_towards_coin == 2, 3, opposite_moves)
        opposite_moves = jnp.where(moves_towards_coin == 3, 2, opposite_moves)

        return opposite_moves

    def get_coop_action(self, state, red_agent_perspective=True):
        # move toward coin if same colour, away if opposite colour
        # An agent that always does this is considered to 'always cooperate'
        moves_towards_coin = self.get_moves_shortest_path_to_coin(state, red_agent_perspective=red_agent_perspective)
        moves_away_from_coin = self.get_moves_away_from_coin(moves_towards_coin)

        coop_moves = jnp.zeros_like(moves_towards_coin) - 1
        if red_agent_perspective:
            is_my_coin = state.is_red_coin
        else:
            is_my_coin = 1 - state.is_red_coin

        is_my_coin = is_my_coin.squeeze(-1)

        coop_moves = jnp.where(is_my_coin == 1, moves_towards_coin, coop_moves)
        coop_moves = jnp.where(is_my_coin == 0, moves_away_from_coin, coop_moves)

        return coop_moves
