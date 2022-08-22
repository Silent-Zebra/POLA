import jax
import jax.numpy as jnp
from typing import NamedTuple
from typing import Tuple


class CoinGameState(NamedTuple):
    red_pos: jnp.ndarray
    blue_pos: jnp.ndarray
    red_coin_pos: jnp.ndarray
    blue_coin_pos: jnp.ndarray
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
    def __init__(self):
        pass

    def reset(self, subkey) -> Tuple[jnp.ndarray, CoinGameState]:
        # print(subkey, flush=True)
        # print("HI", flush=True)
        # exit(-1)
        all_pos = jax.random.randint(subkey, shape=(4, 2), minval=0, maxval=3)
        step_count = jnp.zeros(1)
        state = CoinGameState(all_pos[0], all_pos[1], all_pos[2], all_pos[3], step_count)
        obs = self.state_to_obs(state)
        return state, obs

    def state_to_obs(self, state: CoinGameState) -> jnp.ndarray:
        obs = jnp.zeros((4, 3, 3))
        obs = obs.at[0, state.red_pos[0], state.red_pos[1]].set(1.0)
        obs = obs.at[1, state.blue_pos[0], state.blue_pos[1]].set(1.0)
        obs = obs.at[2, state.red_coin_pos[0], state.red_coin_pos[1]].set(1.0)
        obs = obs.at[3, state.blue_coin_pos[0], state.blue_coin_pos[1]].set(1.0)
        obs = obs.reshape(36)
        return obs

    def step(self, state: CoinGameState, action_0: int, action_1: int, subkey: jnp.ndarray) -> Tuple[jnp.ndarray, list]:
        new_red_pos = (state.red_pos + MOVES[action_0]) % 3
        new_blue_pos = (state.blue_pos + MOVES[action_1]) % 3

        zero_rew = jnp.zeros(1)

        red_red_matches = jnp.all(new_red_pos == state.red_coin_pos, axis=-1)
        red_blue_matches = jnp.all(new_red_pos == state.blue_coin_pos, axis=-1)

        red_reward = jnp.where(red_red_matches, zero_rew + 1, zero_rew)
        red_reward += jnp.where(red_blue_matches, zero_rew + 1, zero_rew)

        blue_red_matches = jnp.all(new_blue_pos == state.red_coin_pos, axis=-1)
        blue_blue_matches = jnp.all(new_blue_pos == state.blue_coin_pos, axis=-1)

        blue_reward = jnp.where(blue_red_matches, zero_rew + 1, zero_rew)
        blue_reward += jnp.where(blue_blue_matches, zero_rew + 1, zero_rew)

        red_reward += jnp.where(blue_red_matches, zero_rew - 2, zero_rew)
        blue_reward += jnp.where(red_blue_matches, zero_rew - 2, zero_rew)


        # TODO import stuff from the coin game modifications in other file, preventing spawn under agents, etc.
        new_random_coin_poses = jax.random.randint(subkey, shape=(2, 2), minval=0, maxval=3)
        new_red_coin_pos = jnp.where(jnp.logical_or(red_red_matches, blue_red_matches), new_random_coin_poses[0], state.red_coin_pos)
        new_blue_coin_pos = jnp.where(jnp.logical_or(red_blue_matches, blue_blue_matches), new_random_coin_poses[1], state.blue_coin_pos)
        step_count = state.step_count + 1

        new_state = CoinGameState(new_red_pos, new_blue_pos, new_red_coin_pos, new_blue_coin_pos, step_count)
        obs = self.state_to_obs(new_state)

        return new_state, obs, (red_reward, blue_reward), (red_red_matches, red_blue_matches, blue_red_matches, blue_blue_matches)
