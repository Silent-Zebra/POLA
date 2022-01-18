# Modified from SSD code repository

import numpy as np

from agent import HarvestAgent  # HARVEST_VIEW_SIZE
from constants import HARVEST_MAP, HARVEST_MAP_BIG, HARVEST_MAP_TINY
from map_env import MapEnv, ACTIONS

APPLE_RADIUS = 2

# Add custom actions to the agent
ACTIONS['FIRE'] = 5  # length of firing range

SPAWN_PROB = [0, 0.005, 0.02, 0.05]


class HarvestEnv(MapEnv):

    def __init__(self, ascii_map=HARVEST_MAP, num_agents=1, render=False, ir_param_list=None,
                         hit_penalty=50, fire_cost=1):
        super().__init__(ascii_map, num_agents, render, ir_param_list=ir_param_list,
                         hit_penalty=hit_penalty, fire_cost=fire_cost)
        # self.hit_penalty = hit_penalty
        # self.fire_cost = fire_cost
        self.apple_points = []
        for row in range(self.base_map.shape[0]):
            for col in range(self.base_map.shape[1]):
                if self.base_map[row, col] == 'A':
                    self.apple_points.append([row, col])

    @property
    def action_space(self):
        agents = list(self.agents.values())
        return agents[0].action_space

    @property
    def observation_space(self):
        agents = list(self.agents.values())
        return agents[0].observation_space

    def setup_agents(self):
        map_with_agents = self.get_map_with_agents()

        for i in range(self.num_agents):
            agent_id = 'agent-' + str(i)
            spawn_point = self.spawn_point()
            rotation = self.spawn_rotation()
            grid = map_with_agents
            if self.ir_param_list is None:
                agent_params=None
            else:
                agent_params = self.ir_param_list[i]
            if agent_params is None:
                agent = HarvestAgent(agent_id, spawn_point, rotation, grid, self.num_agents,
                                     hit_penalty=self.hit_penalty, fire_cost=self.fire_cost)
            elif agent_params[0].lower() == "ineq":
                agent = HarvestAgent(agent_id, spawn_point, rotation, grid,
                                     self.num_agents,
                                     intrinsic_rew_type="ineq",
                                     ineq_alpha=agent_params[1],
                                     ineq_beta=agent_params[2],
                                     rew_scale=agent_params[3],
                                     rew_smoothing=agent_params[4],
                                     hit_penalty=self.hit_penalty, fire_cost=self.fire_cost)
            elif agent_params[0].lower() == "altruism":
                agent = HarvestAgent(agent_id, spawn_point, rotation, grid,
                                     self.num_agents,
                                     intrinsic_rew_type="altruism",
                                     w_self=agent_params[1],
                                     w_others=agent_params[2],
                                     rew_scale=agent_params[3],
                                     rew_smoothing=agent_params[4],
                                     hit_penalty=self.hit_penalty, fire_cost=self.fire_cost)
            elif agent_params[0].lower() == "svo":
                agent = HarvestAgent(agent_id, spawn_point, rotation, grid,
                                     self.num_agents,
                                     intrinsic_rew_type="svo",
                                     svo_angle=agent_params[1],
                                     svo_weight=agent_params[2],
                                     rew_scale=agent_params[3],
                                     rew_smoothing=agent_params[4],
                                     hit_penalty=self.hit_penalty, fire_cost=self.fire_cost)
            elif agent_params[0].lower() == "gini":
                agent = HarvestAgent(agent_id, spawn_point, rotation, grid,
                                     self.num_agents,
                                     intrinsic_rew_type="gini",
                                     gini_weight=agent_params[1],
                                     # svo_weight=agent_params[2],
                                     rew_scale=agent_params[2],
                                     rew_smoothing=agent_params[3],
                                     hit_penalty=self.hit_penalty, fire_cost=self.fire_cost)
            elif agent_params[0].lower() == "vengeance":
                agent = HarvestAgent(agent_id, spawn_point, rotation, grid,
                                     self.num_agents,
                                     intrinsic_rew_type="vengeance",
                                     vengeance_threshold=agent_params[1],
                                     vengeance_rew=agent_params[2],
                                     rew_scale=agent_params[3],
                                     rew_smoothing=agent_params[4],
                                     hit_penalty=self.hit_penalty, fire_cost=self.fire_cost)
            # agent = HarvestAgent(agent_id, spawn_point, rotation, grid)

            # grid = util.return_view(map_with_agents, spawn_point,
            #                         HARVEST_VIEW_SIZE, HARVEST_VIEW_SIZE)
            # agent = HarvestAgent(agent_id, spawn_point, rotation, grid)
            self.agents[agent_id] = agent

    def custom_reset(self):
        """Initialize the walls and the apples"""
        for apple_point in self.apple_points:
            self.world_map[apple_point[0], apple_point[1]] = 'A'

    def custom_action(self, agent, action):
        agent.fire_beam('F')
        updates, agents_hit = self.update_map_fire(agent.get_pos().tolist(),
                                       agent.get_orientation(),
                                       ACTIONS['FIRE'], fire_char='F')

        agent.agents_hit = agents_hit
        return updates

    def custom_map_update(self):
        "See parent class"
        # spawn the apples
        new_apples = self.spawn_apples()
        self.update_map(new_apples)

    def spawn_apples(self):
        """Construct the apples spawned in this step.

        Returns
        -------
        new_apple_points: list of 2-d lists
            a list containing lists indicating the spawn positions of new apples
        """

        new_apple_points = []
        for i in range(len(self.apple_points)):
            row, col = self.apple_points[i]
            # apples can't spawn where agents are standing or where an apple already is
            if [row, col] not in self.agent_pos and self.world_map[row, col] != 'A':
                num_apples = 0
                for j in range(-APPLE_RADIUS, APPLE_RADIUS + 1):
                    for k in range(-APPLE_RADIUS, APPLE_RADIUS + 1):
                        if j**2 + k**2 <= APPLE_RADIUS:
                            x, y = self.apple_points[i]
                            if 0 <= x + j < self.world_map.shape[0] and \
                                    self.world_map.shape[1] > y + k >= 0:
                                symbol = self.world_map[x + j, y + k]
                                if symbol == 'A':
                                    num_apples += 1

                spawn_prob = SPAWN_PROB[min(num_apples, 3)]
                rand_num = np.random.rand(1)[0]
                if rand_num < spawn_prob:
                    new_apple_points.append((row, col, 'A'))
        return new_apple_points

    def count_apples(self, window):
        # compute how many apples are in window
        unique, counts = np.unique(window, return_counts=True)
        counts_dict = dict(zip(unique, counts))
        num_apples = counts_dict.get('A', 0)
        return num_apples
