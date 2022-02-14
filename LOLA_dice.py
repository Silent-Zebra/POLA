import torch
import math
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class CoinGameGPU:
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
        
        red_pos_flat = torch.randint(self.grid_size * self.grid_size, size=(self.batch_size,)).to(device)
        self.red_pos = torch.stack((red_pos_flat // self.grid_size, red_pos_flat % self.grid_size), dim=-1)
        
        blue_pos_flat = torch.randint(self.grid_size * self.grid_size, size=(self.batch_size,)).to(device)
        self.blue_pos = torch.stack((blue_pos_flat // self.grid_size, blue_pos_flat % self.grid_size), dim=-1)

        red_coin_pos_flat = torch.randint(self.grid_size * self.grid_size, size=(self.batch_size,)).to(device)
        blue_coin_pos_flat = torch.randint(self.grid_size * self.grid_size, size=(self.batch_size,)).to(device)
        
        self.red_coin_pos = torch.stack((red_coin_pos_flat // self.grid_size, red_coin_pos_flat % self.grid_size), dim=-1)
        self.blue_coin_pos = torch.stack((blue_coin_pos_flat // self.grid_size, blue_coin_pos_flat % self.grid_size), dim=-1)

        state = self._generate_state()
        observations = [state, state]
        return observations

    def _generate_coins(self):
        mask_red = torch.logical_or(self._same_pos(self.red_coin_pos, self.blue_pos), self._same_pos(self.red_coin_pos, self.red_pos))
        red_coin_pos_flat = torch.randint(self.grid_size * self.grid_size, size=(self.batch_size,)).to(device)[mask_red]
        self.red_coin_pos[mask_red] = torch.stack((red_coin_pos_flat // self.grid_size, red_coin_pos_flat % self.grid_size), dim=-1)        

        mask_blue = torch.logical_or(self._same_pos(self.blue_coin_pos, self.blue_pos), self._same_pos(self.blue_coin_pos, self.red_pos))
        blue_coin_pos_flat = torch.randint(self.grid_size * self.grid_size, size=(self.batch_size,)).to(device)[mask_blue]
        self.blue_coin_pos[mask_blue] = torch.stack((blue_coin_pos_flat // self.grid_size, blue_coin_pos_flat % self.grid_size), dim=-1)        

    def _same_pos(self, x, y):
        return torch.all(x == y, dim=-1)

    def _generate_state(self):
        red_pos_flat = self.red_pos[:,0] * self.grid_size + self.red_pos[:, 1]
        blue_pos_flat = self.blue_pos[:, 0] * self.grid_size + self.blue_pos[:, 1]

        red_coin_pos_flat = self.red_coin_pos[:,0] * self.grid_size + self.red_coin_pos[:,1]
        blue_coin_pos_flat = self.blue_coin_pos[:,0] * self.grid_size + self.blue_coin_pos[:,1]
        
        state = torch.zeros((self.batch_size, 4, self.grid_size*self.grid_size)).to(device)

        state[:,0].scatter_(1, red_pos_flat[:,None], 1)
        state[:,1].scatter_(1, blue_pos_flat[:,None], 1)
        state[:,2].scatter_(1, red_coin_pos_flat[:,None], 1)
        state[:,3].scatter_(1, blue_coin_pos_flat[:,None], 1)


        return state.view(self.batch_size, 4, self.grid_size, self.grid_size)

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
        observations = [state, state]
        if self.step_count >= self.max_steps:
            done = torch.ones(self.batch_size).to(device)
        else:
            done = torch.zeros(self.batch_size).to(device)

        return observations, reward, done, (red_red_matches.sum(), red_blue_matches.sum(), blue_red_matches.sum(), blue_blue_matches.sum())

class Hp():
    def __init__(self):
        self.lr_out = 0.2
        self.lr_in = 0.3
        self.lr_v = 0.1
        self.gamma = 0.96
        self.n_update = 200
        self.len_rollout = 16
        self.batch_size = 256
        self.use_baseline = True
        self.seed = 42
        self.hidden_size = 16

hp = Hp()

env = CoinGameGPU(max_steps=hp.len_rollout, batch_size=hp.batch_size)

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
        cum_discount = torch.cumprod(hp.gamma * torch.ones(*rewards.size()), dim=1).to(device)/hp.gamma
        discounted_rewards = rewards * cum_discount
        discounted_values = values * cum_discount

        # stochastics nodes involved in rewards dependencies:
        dependencies = torch.cumsum(self_logprobs + other_logprobs, dim=1)

        # logprob of each stochastic nodes:
        stochastic_nodes = self_logprobs + other_logprobs

        # dice objective:
        dice_objective = torch.mean(torch.sum(magic_box(dependencies) * discounted_rewards, dim=1))

        if hp.use_baseline:
            # variance_reduction:
            baseline_term = torch.mean(torch.sum((1 - magic_box(stochastic_nodes)) * discounted_values, dim=1))
            dice_objective = dice_objective + baseline_term

        return -dice_objective # want to minimize -objective

    def value_loss(self):
        values = torch.stack(self.values, dim=1)
        rewards = torch.stack(self.rewards, dim=1)
        return torch.mean((rewards - values)**2)

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
    h_v, values = apply(batch_states, theta_v, h_v)
    dist = Categorical(torch.softmax(out, dim=-1))
    actions = dist.sample()
    log_probs_actions = dist.log_prob(actions)
    return actions, log_probs_actions, values.squeeze(-1), h_p, h_v

def get_gradient(objective, theta):
    # create differentiable gradient for 2nd orders:
    grad_objective = torch.autograd.grad(objective, (theta), create_graph=True)
    return grad_objective

def step(theta1, theta2, values1, values2):
    # just to evaluate progress:
    (s1, s2) = env.reset()
    score1 = 0
    score2 = 0
    h_p1, h_v1, h_p2, h_v2 = (torch.zeros(hp.batch_size, hp.hidden_size).to(device), 
                              torch.zeros(hp.batch_size, hp.hidden_size).to(device), 
                              torch.zeros(hp.batch_size, hp.hidden_size).to(device), 
                              torch.zeros(hp.batch_size, hp.hidden_size).to(device))
    for t in range(hp.len_rollout):
        a1, lp1, v1, h_p1, h_v1 = act(s1, theta1, values1, h_p1, h_v1)
        a2, lp2, v2, h_p2, h_v2 = act(s2, theta2, values2, h_p2, h_v2)
        (s1, s2), (r1, r2),_,_ = env.step((a1, a2))
        # cumulate scores
        score1 += torch.mean(r1)/float(hp.len_rollout)
        score2 += torch.mean(r2)/float(hp.len_rollout)
    return (score1, score2)

class Agent():
    def __init__(self, input_size, hidden_size, action_size):
        self.hidden_size = hidden_size
        self.theta_p = nn.ParameterList([
            # Linear 1
            nn.Parameter(torch.zeros((input_size, hidden_size*3), requires_grad=True)),
            nn.Parameter(torch.zeros(hidden_size*3, requires_grad=True)),
            
            # x2h GRU
            nn.Parameter(torch.zeros((hidden_size*3, hidden_size*3), requires_grad=True)),
            nn.Parameter(torch.zeros(hidden_size*3, requires_grad=True)),
            
            # h2h GRU
            nn.Parameter(torch.zeros((hidden_size, hidden_size*3), requires_grad=True)),
            nn.Parameter(torch.zeros(hidden_size*3, requires_grad=True)),
            
            # Linear 2
            nn.Parameter(torch.zeros((hidden_size, action_size), requires_grad=True)),
            nn.Parameter(torch.zeros(action_size, requires_grad=True)),
        ]).to(device)
        
        self.theta_v = nn.ParameterList([
            # Linear 1
            nn.Parameter(torch.zeros((input_size, hidden_size*3), requires_grad=True)),
            nn.Parameter(torch.zeros(hidden_size*3, requires_grad=True)),
            
            # x2h GRU
            nn.Parameter(torch.zeros((hidden_size*3, hidden_size*3), requires_grad=True)),
            nn.Parameter(torch.zeros(hidden_size*3, requires_grad=True)),
            
            # h2h GRU
            nn.Parameter(torch.zeros((hidden_size, hidden_size*3), requires_grad=True)),
            nn.Parameter(torch.zeros(hidden_size*3, requires_grad=True)),
            
            # Linear 2
            nn.Parameter(torch.zeros((hidden_size, 1), requires_grad=True)),
            nn.Parameter(torch.zeros(1, requires_grad=True)),
        ]).to(device)
        
        self.reset_parameters()
        self.theta_optimizer = torch.optim.Adam(self.theta_p,lr=hp.lr_out)
        self.value_optimizer = torch.optim.Adam(self.theta_v,lr=hp.lr_v)

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

    def in_lookahead(self, other_theta, other_values):
        (s1, s2) = env.reset()
        other_memory = Memory()
        h_p1, h_v1, h_p2, h_v2 = (torch.zeros(hp.batch_size, self.hidden_size).to(device), 
                                  torch.zeros(hp.batch_size, self.hidden_size).to(device), 
                                  torch.zeros(hp.batch_size, self.hidden_size).to(device), 
                                  torch.zeros(hp.batch_size, self.hidden_size).to(device))
        
        for t in range(hp.len_rollout):
            a1, lp1, v1, h_p1, h_v1 = act(s1, self.theta_p, self.theta_v, h_p1, h_v1)
            a2, lp2, v2, h_p2, h_v2 = act(s2, other_theta, other_values, h_p2, h_v2)
            (s1, s2), (r1, r2),_,_ = env.step((a1, a2))
            other_memory.add(lp2, lp1, v2, r2)

        other_objective = other_memory.dice_objective()
        grad = get_gradient(other_objective, other_theta)
        return grad

    def out_lookahead(self, other_theta, other_values):
        (s1, s2) = env.reset()
        memory = Memory()
        h_p1, h_v1, h_p2, h_v2 = (torch.zeros(hp.batch_size, self.hidden_size).to(device), 
                                  torch.zeros(hp.batch_size, self.hidden_size).to(device), 
                                  torch.zeros(hp.batch_size, self.hidden_size).to(device), 
                                  torch.zeros(hp.batch_size, self.hidden_size).to(device))
        for t in range(hp.len_rollout):
            a1, lp1, v1, h_p1, h_v1 = act(s1, self.theta_p, self.theta_v, h_p1, h_v1)
            a2, lp2, v2, h_p2, h_v2 = act(s2, other_theta, other_values, h_p2, h_v2)
            (s1, s2), (r1, r2),_,_ = env.step((a1, a2))
            memory.add(lp1, lp2, v1, r1)

        # update self theta
        objective = memory.dice_objective()
        self.theta_update(objective)
        # update self value:
        v_loss = memory.value_loss()
        self.value_update(v_loss)

def play(agent1, agent2, n_lookaheads):
    joint_scores = []
    print("start iterations with", n_lookaheads, "lookaheads:")
    for update in range(hp.n_update):
        # copy other's parameters:
        theta1_ = [torch.tensor(tp.detach(), requires_grad=True) for tp in agent1.theta_p]
        values1_ = [torch.tensor(tv.detach(), requires_grad=True) for tv in agent1.theta_v]
        theta2_ = [torch.tensor(tp.detach(), requires_grad=True) for tp in agent2.theta_p]
        values2_ = [torch.tensor(tv.detach(), requires_grad=True) for tv in agent1.theta_v]

        for k in range(n_lookaheads):
            # estimate other's gradients from in_lookahead:
            grad2 = agent1.in_lookahead(theta2_, values2_)
            grad1 = agent2.in_lookahead(theta1_, values1_)
            # update other's theta
            theta2_ = [theta2_[i] - hp.lr_in * grad2[i] for i in range(len(theta2_))]
            theta1_ = [theta1_[i] - hp.lr_in * grad1[i] for i in range(len(theta1_))]

        # update own parameters from out_lookahead:
        agent1.out_lookahead(theta2_, values2_)
        agent2.out_lookahead(theta1_, values1_)

        # evaluate progress:
        score = step(agent1.theta_p, agent2.theta_p, agent1.theta_v, agent2.theta_v)
        joint_scores.append(0.5*(score[0] + score[1]))

        # print
        if update%10==0 :
#             p1 = [p.item() for p in torch.sigmoid(agent1.theta)]
#             p2 = [p.item() for p in torch.sigmoid(agent2.theta)]
#             print('update', update, 'score (%.3f,%.3f)' % (score[0], score[1]) , 'policy (agent1) = {%.3f, %.3f, %.3f, %.3f, %.3f}' % (p1[0], p1[1], p1[2], p1[3], p1[4]),' (agent2) = {%.3f, %.3f, %.3f, %.3f, %.3f}' % (p2[0], p2[1], p2[2], p2[3], p2[4]))
            print("*"*10)
            print(f"Score 0: {score[0]}")
            print(f"Score 1: {score[1]}")

    return joint_scores

if __name__ == "__main__":
    i = 1
    input_size = 36
    action_size = 4
    scores = play(Agent(input_size, hp.hidden_size, action_size), Agent(input_size, hp.hidden_size, action_size), i)