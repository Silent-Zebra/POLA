import torch
import math
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
import argparse

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
        red_red_matches.sum(), red_blue_matches.sum(), blue_red_matches.sum(),
        blue_blue_matches.sum())



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
    for t in range(args.len_rollout):
        a1, lp1, v1, h_p1, h_v1, cat_act_probs1 = act(s1, theta1, values1, h_p1, h_v1)
        a2, lp2, v2, h_p2, h_v2, cat_act_probs2 = act(s2, theta2, values2, h_p2, h_v2)
        (s1, s2), (r1, r2), _, info = env.step((a1, a2))
        # cumulate scores
        score1 += torch.mean(r1) / float(args.len_rollout)
        score2 += torch.mean(r2) / float(args.len_rollout)
        # print(info)

    return (score1, score2), info


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
        self.theta_optimizer = torch.optim.Adam(self.theta_p, lr=args.lr_out)
        self.value_optimizer = torch.optim.Adam(self.theta_v, lr=args.lr_v)

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

    def in_lookahead(self, other_theta, other_values):
        (s1, s2) = env.reset()
        other_memory = Memory()
        h_p1, h_v1, h_p2, h_v2 = (
        torch.zeros(args.batch_size, self.hidden_size).to(device),
        torch.zeros(args.batch_size, self.hidden_size).to(device),
        torch.zeros(args.batch_size, self.hidden_size).to(device),
        torch.zeros(args.batch_size, self.hidden_size).to(device))

        for t in range(args.len_rollout):
            a1, lp1, v1, h_p1, h_v1, cat_act_probs1 = act(s1, self.theta_p, self.theta_v, h_p1,
                                          h_v1)
            a2, lp2, v2, h_p2, h_v2, cat_act_probs2 = act(s2, other_theta, other_values, h_p2,
                                          h_v2)
            (s1, s2), (r1, r2), _, _ = env.step((a1, a2))
            other_memory.add(lp2, lp1, v2, r2)

        other_objective = other_memory.dice_objective()
        grad = get_gradient(other_objective, other_theta)
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

        if not first_outer_step:
            curr_pol_probs = self.get_policies_for_states()
            kl_div = torch.nn.functional.kl_div(torch.log(curr_pol_probs), self.ref_cat_act_probs.detach(), log_target=False, reduction='batchmean')
            print(kl_div)

        # update self theta
        objective = memory.dice_objective()
        if not first_outer_step:
            objective += args.outer_beta * kl_div
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

            for k in range(n_lookaheads):
                # estimate other's gradients from in_lookahead:
                grad2 = agent1.in_lookahead(theta2_, values2_)
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

            for k in range(n_lookaheads):
                # estimate other's gradients from in_lookahead:
                grad1 = agent2.in_lookahead(theta1_, values1_)
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
        same_colour_coins = rr_matches + bb_matches
        diff_colour_coins = rb_matches + br_matches
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
    parser.add_argument("--n_update", type=int, default=1000, help="number of epochs to run")
    parser.add_argument("--len_rollout", type=int, default=50, help="How long we want the time horizon of the game to be (number of steps before termination/number of iterations of the IPD)")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1, help="for seed")
    parser.add_argument("--hidden_size", type=int, default=16)
    parser.add_argument("--print_every", type=int, default=1, help="Print every x number of epochs")
    parser.add_argument("--outer_beta", type=float, default=0.0, help="for outer kl penalty with POLA")

    use_baseline = True

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    input_size = 36
    action_size = 4

    env = CoinGameGPU(max_steps=args.len_rollout, batch_size=args.batch_size)

    scores = play(Agent(input_size, args.hidden_size, action_size),
                  Agent(input_size, args.hidden_size, action_size),
                  args.inner_steps, args.outer_steps)
