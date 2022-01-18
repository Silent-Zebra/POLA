# Modified from SSD code repository


"""Base class for an agent that defines the possible actions. """

from gym.spaces import Box, Tuple
from gym.spaces import Discrete
import numpy as np
import utility_funcs as util

# basic moves every agent should do
BASE_ACTIONS = {0: 'MOVE_LEFT',  # Move left
                1: 'MOVE_RIGHT',  # Move right
                2: 'MOVE_UP',  # Move up
                3: 'MOVE_DOWN',  # Move down
                4: 'STAY',  # don't move
                5: 'TURN_CLOCKWISE',  # Rotate counter clockwise
                6: 'TURN_COUNTERCLOCKWISE'}  # Rotate clockwise


class Agent(object):

    def __init__(self, agent_id, start_pos, start_orientation, grid, row_size, col_size,
                 intrinsic_rew_type=None, ineq_alpha=None, ineq_beta=None, w_self=None, w_others=None,
                 svo_angle=None, svo_weight=None, gini_weight=None,
                 vengeance_threshold=None, vengeance_rew=None, rew_scale=1.0, rew_smoothing=0.0):
        """Superclass for all agents.

        Parameters
        ----------
        agent_id: (str)
            a unique id allowing the map to identify the agents
        start_pos: (np.ndarray)
            a 2d array indicating the x-y position of the agents
        start_orientation: (np.ndarray)
            a 2d array containing a unit vector indicating the agent direction
        grid: (2d array)
            a reference to this agent's view of the environment
        row_size: (int)
            how many rows up and down the agent can look
        col_size: (int)
            how many columns left and right the agent can look
        """
        self.ineq_alpha = ineq_alpha
        self.ineq_beta = ineq_beta
        self.w_self = w_self
        self.w_others = w_others
        self.svo_angle = svo_angle
        self.svo_weight = svo_weight
        self.gini_weight = gini_weight
        self.vengeance_threshold = vengeance_threshold
        self.vengeance_rew = vengeance_rew
        self.rew_scale = rew_scale # Scaling for turning down effective learning rate on intrinsic reward. A multiplicative factor (so 0.01 to reduce learning by factor of 100)
        self.rew_smoothing = rew_smoothing # I formerly called lambdgamma, this is for exponential smoothing of rewards, introduced in inequity aversion paper, intended to help credit assignment

        self.times_hit = 0
        self.fires = 0

        self.agent_id = agent_id
        self.pos = np.array(start_pos)
        self.orientation = start_orientation
        self.intrinsic_rew_type = intrinsic_rew_type
        # TODO(ev) change grid to env, this name is not very informative
        self.grid = grid
        self.row_size = row_size
        self.col_size = col_size
        self.reward_this_turn = 0

    @property
    def action_space(self):
        """Identify the dimensions and bounds of the action space.

        MUST BE implemented in new environments.

        Returns
        -------
        gym Box, Discrete, or Tuple type
            a bounded box depicting the shape and bounds of the action space
        """
        raise NotImplementedError

    @property
    def observation_space(self):
        """Identify the dimensions and bounds of the observation space.

        MUST BE implemented in new environments.

        Returns
        -------
        gym Box, Discrete or Tuple type
            a bounded box depicting the shape and bounds of the observation
            space
        """
        raise NotImplementedError

    def action_map(self, action_number):
        """Maps action_number to a desired action in the map"""
        raise NotImplementedError

    def get_state(self):
        return util.return_view(self.grid, self.get_pos(),
                                self.row_size, self.col_size)

    def compute_reward(self):
        reward = self.reward_this_turn
        self.reward_this_turn = 0
        return reward

    def set_pos(self, new_pos):
        self.pos = np.array(new_pos)

    def get_pos(self):
        return self.pos

    def translate_pos_to_egocentric_coord(self, pos):
        offset_pos = pos - self.get_pos()
        ego_centre = [self.row_size, self.col_size]
        return ego_centre + offset_pos

    def set_orientation(self, new_orientation):
        self.orientation = new_orientation

    def get_orientation(self):
        return self.orientation

    def get_map(self):
        return self.grid

    def return_valid_pos(self, new_pos):
        """Checks that the next pos is legal, if not return current pos"""
        ego_new_pos = new_pos  # self.translate_pos_to_egocentric_coord(new_pos)
        new_row, new_col = ego_new_pos
        # you can't walk through walls
        temp_pos = new_pos.copy()
        if self.grid[new_row, new_col] == '@':
            temp_pos = self.get_pos()
        return temp_pos

    def update_agent_pos(self, new_pos):
        """Updates the agents internal positions

        Returns
        -------
        old_pos: (np.ndarray)
            2 element array describing where the agent used to be
        new_pos: (np.ndarray)
            2 element array describing the agent positions
        """
        old_pos = self.get_pos()
        ego_new_pos = new_pos  # self.translate_pos_to_egocentric_coord(new_pos)
        new_row, new_col = ego_new_pos
        # you can't walk through walls
        temp_pos = new_pos.copy()
        if self.grid[new_row, new_col] == '@':
            temp_pos = self.get_pos()
        self.set_pos(temp_pos)
        # TODO(ev) list array consistency
        return self.get_pos(), np.array(old_pos)

    def update_agent_rot(self, new_rot):
        self.set_orientation(new_rot)

    def hit(self, char):
        """Defines how an agent responds to being hit by a beam of type char"""
        raise NotImplementedError

    def consume(self, char):
        """Defines how an agent interacts with the char it is standing on"""
        raise NotImplementedError


HARVEST_ACTIONS = BASE_ACTIONS.copy()
HARVEST_ACTIONS.update({7: 'FIRE'})  # Fire a penalty beam

HARVEST_VIEW_SIZE = 7


class HarvestAgent(Agent):

    def __init__(self, agent_id, start_pos, start_orientation, grid, num_agents,
                 view_len=HARVEST_VIEW_SIZE, intrinsic_rew_type=None,
                 ineq_alpha=None, ineq_beta=None, w_self=None, w_others=None,
                 svo_angle=None, svo_weight=None, gini_weight=None, vengeance_threshold=None,
                 vengeance_rew=None, rew_scale=1.0, rew_smoothing=0.0, hit_penalty=50, fire_cost=1):
        self.hit_penalty = hit_penalty
        self.fire_cost = fire_cost
        self.view_len = view_len
        self.num_agents = num_agents
        super().__init__(agent_id, start_pos, start_orientation, grid, view_len, view_len,
                         intrinsic_rew_type, ineq_alpha, ineq_beta, w_self, w_others,
                         svo_angle, svo_weight, gini_weight, vengeance_threshold,
                         vengeance_rew, rew_scale, rew_smoothing)
        self.update_agent_pos(start_pos)
        self.update_agent_rot(start_orientation)
        self.agents_hit = None

    @property
    def action_space(self):
        return Discrete(8)

    # Ugh, this is gross, this leads to the actions basically being
    # defined in two places
    def action_map(self, action_number):
        """Maps action_number to a desired action in the map"""
        return HARVEST_ACTIONS[action_number]

    @property
    def observation_space(self):
        map_obs = Box(low=0.0, high=255.0, shape=(2 * self.view_len + 1,
                                             2 * self.view_len + 1, 3), dtype=np.float32)
        # if self.intrinsic_rew_type is None:
        #     return map_obs
        # else:
        rew_obs = Box(low=-np.inf, high=np.inf, shape=(self.num_agents,), dtype=np.float32)
        return Tuple([map_obs, rew_obs])
    # def observation_space(self):
    #     return Box(low=0.0, high=0.0, shape=(2 * self.view_len + 1,
    #                                          2 * self.view_len + 1, 3), dtype=np.float32)

    def hit(self, char):
        if char == 'F':
            self.reward_this_turn -= self.hit_penalty
            self.times_hit += 1

    def fire_beam(self, char):
        if char == 'F':
            self.reward_this_turn -= self.fire_cost
            self.fires += 1

    def get_done(self):
        return False

    def consume(self, char):
        """Defines how an agent interacts with the char it is standing on"""
        if char == 'A':
            self.reward_this_turn += 1
            return ' '
        else:
            return char


CLEANUP_ACTIONS = BASE_ACTIONS.copy()
CLEANUP_ACTIONS.update({7: 'FIRE',  # Fire a penalty beam
                        8: 'CLEAN'})  # Fire a cleaning beam

CLEANUP_VIEW_SIZE = 7


class CleanupAgent(Agent):
    def __init__(self, agent_id, start_pos, start_orientation, grid, num_agents,
                 view_len=CLEANUP_VIEW_SIZE, intrinsic_rew_type=None,
                 ineq_alpha=None, ineq_beta=None, w_self=None, w_others=None,
                 svo_angle=None, svo_weight=None, gini_weight=None, vengeance_threshold=None,
                 vengeance_rew=None, rew_scale=1.0, rew_smoothing=0.0, hit_penalty=50, fire_cost=1):

        self.hit_penalty = hit_penalty
        self.fire_cost = fire_cost
        self.view_len = view_len
        self.num_agents = num_agents
        self.cleans = 0
        super().__init__(agent_id, start_pos, start_orientation, grid, view_len,
                         view_len, intrinsic_rew_type, ineq_alpha, ineq_beta, w_self, w_others,
                         svo_angle, svo_weight, gini_weight, vengeance_rew, vengeance_threshold,
                         rew_scale, rew_smoothing)
        # remember what you've stepped on
        self.update_agent_pos(start_pos)
        self.update_agent_rot(start_orientation)

    @property
    def action_space(self):
        return Discrete(9)

    @property
    def observation_space(self):
        map_obs = Box(low=0.0, high=255.0, shape=(2 * self.view_len + 1,
                                                2 * self.view_len + 1, 3),
                      dtype=np.float32)
        # if self.intrinsic_rew_type is None:
        #     return map_obs
        # else:
        rew_obs = Box(low=-np.inf, high=np.inf, shape=(self.num_agents,),
                      dtype=np.float32)
        return Tuple([map_obs, rew_obs])
        # return Box(low=0.0, high=0.0, shape=(2 * self.view_len + 1,
        #                                      2 * self.view_len + 1, 3), dtype=np.float32)

    # Ugh, this is gross, this leads to the actions basically being
    # defined in two places
    def action_map(self, action_number):
        """Maps action_number to a desired action in the map"""
        return CLEANUP_ACTIONS[action_number]

    def fire_beam(self, char):
        if char == 'F':
            self.reward_this_turn -= self.fire_cost
            self.fires += 1
        if char == 'C':
            self.cleans += 1

    def get_done(self):
        return False

    def hit(self, char):
        if char == 'F':
            self.reward_this_turn -= self.hit_penalty
            self.times_hit += 1

    def consume(self, char):
        """Defines how an agent interacts with the char it is standing on"""
        if char == 'A':
            self.reward_this_turn += 1
            return ' '
        else:
            return char
