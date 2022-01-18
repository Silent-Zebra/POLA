# Modified from SSD code repository

"""Defines a multi-agent controller to rollout environment episodes w/
   agent policies."""

import utility_funcs
import numpy as np
import os
import sys
import shutil
# import tensorflow as tf

from cleanup import CleanupEnv
from harvest import HarvestEnv

from constants import HARVEST_MAP, HARVEST_MAP_BIG, \
    HARVEST_MAP_TINY, HARVEST_MAP_TOY, HARVEST_MAP_CPR, \
    CLEANUP_MAP, CLEANUP_MAP_SMALL

from DQN import DQNAgent, ConvFC


import torch


# FLAGS = tf.app.flags.FLAGS
#
# tf.app.flags.DEFINE_string(
#     'vid_path', os.path.abspath(os.path.join(os.path.dirname(__file__), './videos')),
#     'Path to directory where videos are saved.')
# tf.app.flags.DEFINE_string(
#     'env', 'cleanup',
#     'Name of the environment to rollout. Can be cleanup or harvest.')
# tf.app.flags.DEFINE_string(
#     'render_type', 'pretty',
#     'Can be pretty or fast. Implications obvious.')
# tf.app.flags.DEFINE_integer(
#     'fps', 8,
#     'Number of frames per second.')


def reshape_obs_for_convfc(obs_agent_i):
    return obs_agent_i.reshape(
        obs_agent_i.shape[2], obs_agent_i.shape[0], obs_agent_i.shape[1])


class Controller(object):

    def __init__(self, env_name='harvest', num_agents=1):
        self.env_name = env_name
        if env_name == 'harvest':
            print('Initializing Harvest environment')
            self.env = HarvestEnv(ascii_map=HARVEST_MAP_CPR, num_agents=num_agents, render=True)
        elif env_name == 'cleanup':
            print('Initializing Cleanup environment')
            self.env = CleanupEnv(num_agents=num_agents, render=True)
        else:
            print('Error! Not a valid environment type')
            return

        self.num_agents = num_agents

        self.agent_policies = []
        self.agents = list(self.env.agents.values())
        # print(agents[0].action_space)
        self.action_dim = self.agents[0].action_space.n
        for _ in range(num_agents):
            # TODO right now only using 1 frame, update later to look back x (e.g. 4) frames. Later RNN/LSTM
            neural_net = ConvFC(conv_in_channels=3, # harvest specific input is 15x15x3 (HARVEST_VIEW_SIZE = 7)
                                conv_out_channels=3,
                                input_size=15,
                                hidden_size=64,
                                output_size=self.action_dim)
            self.agent_policies.append(DQNAgent(0, self.action_dim - 1, neural_net))

        self.env.reset()

    def process_experiences(self, id, i, obs, action_dict, rew, next_obs, dones, train_agents=False):
        # print(id)
        # print(i)
        agent_i = "agent-{}".format(i)
        self.agent_policies[i].push_experience(
            reshape_obs_for_convfc(obs[agent_i][0]),
            action_dict[agent_i],
            rew[agent_i], reshape_obs_for_convfc(next_obs[agent_i][0]), # we here using without the reward info... can modify later but this is just a test
            dones[agent_i])

        if train_agents:
            self.agent_policies[i].q_learn_update()

    # def train_parallel_agents(self, id, obs, action_dict, rew, next_obs, dones):
    #     for i in range(self.num_agents):
    #         # torch.multiprocessing.spawn(self.train_agent, args=(i, obs, action_dict, rew, next_obs, dones))
    #         self.train_agent(id, i, obs, action_dict, rew, next_obs, dones)

    def rollout(self, horizon, train_every=100, save_path=None, train_agents=True, print_act=False):
        """ Rollout several timesteps of an episode of the environment.

        Args:
            horizon: The number of timesteps to roll out.
            save_path: If provided, will save each frame to disk at this
                location.
        """


        rewards = np.zeros(self.num_agents)
        observations = []
        shape = self.env.world_map.shape
        full_obs = [np.zeros(
            (shape[0], shape[1], 3), dtype=np.uint8) for i in range(horizon)]

        init_obs = self.env.reset()
        # print(init_obs)
        obs = init_obs

        for time_step in range(horizon):
            # print(time_step )
            action_dim = self.action_dim

            # Single agent hardcoded for now

            hard_coded=True
            if hard_coded:
                action_cycle = 40
                prep_time = 4 + 2 #10
                single_obs = obs["agent-{}".format(0)][0]
                if time_step < prep_time - 2:
                    # print(single_obs)
                    # print(single_obs.shape)
                    # print(single_obs[7][7])
                    #
                    # print(single_obs[7][6])
                    # print(single_obs[6][7])
                    # print(single_obs[7][8])
                    # print(single_obs[8][7])
                    # if single_obs[8][7].sum() == 540 and single_obs[7][6].sum() == 540: # 200
                    if single_obs[6][7].sum() == 540 and single_obs[7][8].sum() == 540: # 200
                    # if single_obs[6][7].sum() == 540 and single_obs[7][6].sum() == 540: # 100
                    # if single_obs[8][7].sum() == 540 and single_obs[7][8].sum() == 540: # 100
                        action = 4
                    # elif single_obs[7][9].sum() == 0 and single_obs[5][7].sum() == 0: # lower and left empty
                    #     action = 5
                    else: action = 6 # got lazy, just keep turning otherwise
                    # action = 5
                # elif time_step == prep_time - 3:
                #     # print(single_obs[7][6])
                #     # print(single_obs[6][7])
                #     # print(single_obs[7][8])
                #     # print(single_obs[8][7])
                #     action=2 # first up movement, start the cycle
                elif time_step == prep_time - 2:
                    # print(single_obs[7][6])
                    # print(single_obs[6][7])
                    # print(single_obs[7][8])
                    # print(single_obs[8][7])
                    action= 1 #0 # first left movement, start the cycle # left and right are wrong? Yeah they messed it up
                    # Um anyway... around 450 is optimal in this env.
                elif time_step == prep_time - 1:
                    # print(single_obs[7][6])
                    # print(single_obs[6][7])
                    # print(single_obs[7][8])
                    # print(single_obs[8][7])
                    action = 2  # up again for smoe reason
                else:
                    # if time_step == prep_time:
                    # print(single_obs[7][6])
                    # print(single_obs[6][7])
                    # print(single_obs[7][8])
                    # print(single_obs[8][7])
                    # Assumes up orientation
                    if (time_step-prep_time) % action_cycle < 16:
                        action = 1 # left
                    elif (time_step-prep_time)  % action_cycle < 20:
                        action = 2
                    elif (time_step-prep_time)  % action_cycle < 36:
                        action = 0 # right
                    elif (time_step-prep_time) % action_cycle < 40:
                        action = 3 # down
                    # print(action)

                actions = [ action ]


            action_dict = {}

            if not hard_coded:
                actions = []
                if train_agents:
                    # for i in range(self.num_agents):
                    #     print(i)
                    #     action = self.agent_policies[i].act(reshape_obs_for_convfc(obs["agent-{}".format(i)]), print_act=print_act)
                    # actions.append(action)
                    actions = [self.agent_policies[i].act(reshape_obs_for_convfc(obs["agent-{}".format(i)][0]), print_act=print_act) for i in range(self.num_agents)]
                else:
                    # can choose eps=0 or something else after
                    actions = [self.agent_policies[i].act(reshape_obs_for_convfc(obs["agent-{}".format(i)][0]), print_act=print_act) for i in range(self.num_agents)]

            for i in range(self.num_agents):
                agent_i = "agent-{}".format(i)
                action_dict[agent_i] = actions[i]
                # if train_agents:
                #     # print(ray.get(self.agent_policies[i].act.remote(reshape_obs_for_convfc(obs[agent_i]))))
                #     action_dict[agent_i] = self.agent_policies[i].act.remote(reshape_obs_for_convfc(obs[agent_i]))
                # else:
                #     action_dict[agent_i] = self.agent_policies[i].act.remote(reshape_obs_for_convfc(obs[agent_i]), epsilon=0)
                #     # 1, obs[agent_i].shape[2], obs[agent_i].shape[0], obs[agent_i].shape[1] )) # batch size = 1 for 1 obs right now...


            next_obs, rew, dones, info, = self.env.step(action_dict)

            if not hard_coded:
                if train_agents:
                    for i in range(self.num_agents):
                        if ((time_step + 1) % train_every == 0):
                            self.process_experiences(0, i, obs, action_dict, rew, next_obs, dones, train_agents=True)
                        else:
                            self.process_experiences(0, i, obs, action_dict, rew, next_obs, dones, train_agents=False)

            obs = next_obs

            sys.stdout.flush()

            if save_path is not None:
                self.env.render(filename=save_path + 'frame' + str(time_step).zfill(6) + '.png')

            rgb_arr = self.env.map_to_colors()
            full_obs[time_step] = rgb_arr.astype(np.uint8)

            # rewards.append(rew)
            observations.append(obs)
            for i in range(self.num_agents):
                agent_i = "agent-{}".format(i)
                rewards[i] += rew[agent_i]
            # observations.append(obs['agent-0'])
            # rewards.append(rew['agent-0'])


        return rewards, observations, full_obs

    def render_rollout(self, horizon=50, path=None,
                       fps=8):
        """ Render a rollout into a video.

        Args:
            horizon: The number of timesteps to roll out.
            path: Directory where the video will be saved.
            render_type: Can be 'pretty' or 'fast'. Impliciations obvious.
            fps: Integer frames per second.
        """
        if path is None:
            path = os.path.abspath(os.path.dirname(__file__)) + '/videos'
            print(path)
            if not os.path.exists(path):
                os.makedirs(path)
        video_name = self.env_name + '_trajectory'

        # if render_type == 'pretty':
        #     image_path = os.path.join(path, 'frames/')
        #     if not os.path.exists(image_path):
        #         os.makedirs(image_path)
        #
        #     rewards, observations, full_obs = self.rollout(
        #         horizon=horizon, save_path=image_path, train_agents=False)
        #     utility_funcs.make_video_from_image_dir(path, image_path, fps=fps,
        #                                             video_name=video_name)
        #
        #     # Clean up images
        #     shutil.rmtree(image_path)
        # else:
        rewards, observations, full_obs = self.rollout(horizon=horizon, train_agents=False, print_act=False)
        utility_funcs.make_video_from_rgb_imgs(full_obs, path, fps=fps,
                                               video_name=video_name)
        return rewards


# def main(unused_argv):
#     c = Controller(env_name=FLAGS.env)
#     c.render_rollout(path=FLAGS.vid_path, render_type=FLAGS.render_type,
#                      fps=FLAGS.fps)
#
#
# if __name__ == '__main__':
#     tf.app.run(main)
