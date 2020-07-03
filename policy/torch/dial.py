# Import libraries
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_

import matplotlib.pyplot as plt

class SwitchGame:
    def __init__(self, opt):
        """
        Initializes the Switch Game with given parameters.
        """     
        self.game_actions = DotDic({
            'NOTHING': 1,
            'TELL': 2
        })

        self.game_states = DotDic({
            'OUTSIDE': 0,
            'INSIDE': 1,
        })

        self.opt = opt

        # Set game defaults
        opt_game_default = DotDic({
            'game_action_space': 2,
            'game_reward_shift': 0,
            'game_comm_bits': 1,
            'game_comm_sigma': 2
        })
        for k in opt_game_default:
            if k not in self.opt:
                self.opt[k] = opt_game_default[k]

        self.opt.nsteps = 4 * self.opt.game_nagents - 6

        self.reward_all_live = 1
        self.reward_all_die = -1

        self.reset()

    def reset(self):
        """
        Resets the environment for the next episode and sets up the agent sequence for the next episode. 
        """
        # Step count
        self.step_count = 0

        # Rewards
        self.reward = torch.zeros(self.opt.bs, self.opt.game_nagents)

        # Who has been in the room?
        self.has_been = torch.zeros(self.opt.bs, self.opt.nsteps, self.opt.game_nagents)

        # Terminal state
        self.terminal = torch.zeros(self.opt.bs, dtype=torch.long)

        # Active agent
        self.active_agent = torch.zeros(self.opt.bs, self.opt.nsteps, dtype=torch.long) # 1-indexed agents
        for b in range(self.opt.bs):
            for step in range(self.opt.nsteps):
                agent_id = 1 + np.random.randint(self.opt.game_nagents)
                self.active_agent[b][step] = agent_id
                self.has_been[b][step][agent_id - 1] = 1

        return self

    def get_action_range(self, step, agent_id):
        """
        Return 1-indexed indices into Q vector for valid actions and communications (so 0 represents no-op)
        """
        opt = self.opt
        action_dtype = torch.long
        action_range = torch.zeros((self.opt.bs, 2), dtype=action_dtype)
        comm_range = torch.zeros((self.opt.bs, 2), dtype=action_dtype)
        for b in range(self.opt.bs): 
            if self.active_agent[b][step] == agent_id:
                action_range[b] = torch.tensor([1, opt.game_action_space], dtype=action_dtype)
                comm_range[b] = torch.tensor(
                    [opt.game_action_space + 1, opt.game_action_space_total], dtype=action_dtype)
            else:
                action_range[b] = torch.tensor([1, 1], dtype=action_dtype)

        return action_range, comm_range

    def get_comm_limited(self, step, agent_id):
        """
        Returns the possible communication options.
        """
        if self.opt.game_comm_limited:
            comm_lim = torch.zeros(self.opt.bs, dtype=torch.long)
            for b in range(self.opt.bs):
                if step > 0 and agent_id == self.active_agent[b][step]:
                    comm_lim[b] = self.active_agent[b][step - 1]
            return comm_lim
        return None

    def get_reward(self, a_t):
        """
        Returns the reward for action a_t taken by current agent in state a_t
        """
        for b in range(self.opt.bs):
            active_agent_idx = self.active_agent[b][self.step_count].item() - 1
            if a_t[b][active_agent_idx].item() == self.game_actions.TELL and not self.terminal[b].item():
                has_been = self.has_been[b][:self.step_count + 1].sum(0).gt(0).sum(0).item()
                if has_been == self.opt.game_nagents:
                    self.reward[b] = self.reward_all_live
                else:
                    self.reward[b] = self.reward_all_die
                self.terminal[b] = 1
            elif self.step_count == self.opt.nsteps - 1 and not self.terminal[b]:
                self.terminal[b] = 1

        return self.reward.clone(), self.terminal.clone()

    def step(self, a_t):
        """
        Executes action a_t by current agent and returns reward and terminal status.
        """
        reward, terminal = self.get_reward(a_t)
        self.step_count += 1

        return reward, terminal

    def get_state(self):
        """
        Returns the current game state.
        """
        state = torch.zeros(self.opt.bs, self.opt.game_nagents, dtype=torch.long)

        # Get the state of the game
        for b in range(self.opt.bs):
            for a in range(1, self.opt.game_nagents + 1):
                if self.active_agent[b][self.step_count] == a:
                    state[b][a - 1] = self.game_states.INSIDE

        return state

    def oracle_strategy_reward(self, steps):
        """
        Returns the episodic return for the optimal strategy, to normalize the rewards.
        """
        reward = torch.zeros(self.opt.bs)
        for b in range(self.opt.bs):
            has_been = self.has_been[b][:self.opt.nsteps].sum(0).gt(0).sum().item()
            if has_been == self.opt.game_nagents:
                reward[b] = self.reward_all_live

        return reward
        
    def get_stats(self, steps):
        stats = DotDic({})
        stats.oracle_reward = self.oracle_strategy_reward(steps)
        return stats

    def describe_game(self, b=0):
        print('has been:', self.has_been[b])
        print('num has been:', self.has_been[b].sum(0).gt(0).sum().item())
        print('active agents: ', self.active_agent[b])
        print('reward:', self.reward[b])

# Dictionary with items accessible as attributes (through '.' operator)
class DotDic(dict):
	__getattr__ = dict.get
	__setattr__ = dict.__setitem__
	__delattr__ = dict.__delitem__

	def __deepcopy__(self, memo=None):
		return DotDic(copy.deepcopy(dict(self), memo=memo))

# To reset the weights of layers in nn.Sequential model.
def weight_reset(m):
    if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.Linear):
        m.reset_parameters()


class DRU:
    def __init__(self, sigma, comm_narrow=True, hard=False):
        self.sigma = sigma
        self.comm_narrow = comm_narrow
        self.hard = hard

    def regularize(self, m):
        """
        Returns the regularized value of message `m` during training.
        """
        m_reg = m + torch.randn(m.size()) * self.sigma
        if self.comm_narrow:
            m_reg = torch.sigmoid(m_reg)
        else:
            m_reg = torch.softmax(m_reg, 0)
        return m_reg

    def discretize(self, m):
        """
        Returns the discretized value of message `m` during execution.
        """
        if self.hard:
            if self.comm_narrow:
                return (m.gt(0.5).float() - 0.5).sign().float()
            else:
                m_ = torch.zeros_like(m)
                if m.dim() == 1:      
                    _, idx = m.max(0)
                    m_[idx] = 1.
                elif m.dim() == 2:      
                    _, idx = m.max(1)
                    for b in range(idx.size(0)):
                        m_[b, idx[b]] = 1.
                else:
                    raise ValueError('Wrong message shape: {}'.format(m.size()))
                return m_
        else:
            scale = 2 * 20
            if self.comm_narrow:
                return torch.sigmoid((m.gt(0.5).float() - 0.5) * scale)
            else:
                return torch.softmax(m * scale, -1)

    def forward(self, m, train_mode):
        if train_mode:
            return self.regularize(m)
        else:
            return self.discretize(m)

class CNet(nn.Module):
    def __init__(self, opts):
        """
        Initializes the CNet model
        """
        super(CNet, self).__init__()
        self.opts = opts
        self.comm_size = opts['game_comm_bits']
        self.init_param_range = (-0.08, 0.08)

        ## Lookup tables for the state, action and previous action.
        self.action_lookup = nn.Embedding(opts['game_nagents'], opts['rnn_size'])
        self.state_lookup = nn.Embedding(2, opts['rnn_size'])
        self.prev_action_lookup = nn.Embedding(opts['game_action_space_total'], opts['rnn_size'])

        # Single layer MLP(with batch normalization for improved performance) for producing embeddings for messages.
        self.message = nn.Sequential(
            nn.BatchNorm1d(self.comm_size),
            nn.Linear(self.comm_size, opts['rnn_size']),
            nn.ReLU(inplace=True)
        )

        # RNN to approximate the agent’s action-observation history.
        self.rnn = nn.GRU(input_size=opts['rnn_size'], hidden_size=opts['rnn_size'], num_layers=2, batch_first=True)

        # 2 layer MLP with batch normalization, for producing output from RNN top layer.
        self.output = nn.Sequential(
            nn.Linear(opts['rnn_size'], opts['rnn_size']),
            nn.BatchNorm1d(opts['rnn_size']),
            nn.ReLU(),
            nn.Linear(opts['rnn_size'], opts['game_action_space_total'])
        )
    
    def get_params(self):
        return list(self.parameters())

    def reset_parameters(self):
        """
        Reset all model parameters
        """
        self.rnn.reset_parameters()
        self.action_lookup.reset_parameters()
        self.state_lookup.reset_parameters()
        self.prev_action_lookup.reset_parameters()
        self.message.apply(weight_reset)
        self.output.apply(weight_reset)
        for p in self.rnn.parameters():
            p.data.uniform_(*self.init_param_range)

    def forward(self, state, messages, hidden, prev_action, agent):
        """
        Returns the q-values and hidden state for the given step parameters
        """
        state = Variable(torch.LongTensor(state))
        hidden = Variable(torch.FloatTensor(hidden))
        prev_action = Variable(torch.LongTensor(prev_action))
        agent = Variable(torch.LongTensor(agent))

        # Produce embeddings for rnn from input parameters
        z_a = self.action_lookup(agent)
        z_o = self.state_lookup(state)
        z_u = self.prev_action_lookup(prev_action)
        z_m = self.message(messages.view(-1, self.comm_size))

        # Add the input embeddings to calculate final RNN input.
        z = z_a + z_o + z_u + z_m
        z = z.unsqueeze(1)

        rnn_out, h = self.rnn(z, hidden)
        # Produce final CNet output q-values from GRU output.
        out = self.output(rnn_out[:, -1, :].squeeze())

        return h, out

class Agent:
    def __init__(self, opts, game, model, target, agent_no):
        """
        Initializes the agent(with id=agent_no) with given model and target_model
        """
        self.game = game
        self.opts = opts
        self.model = model
        self.model_target = target
        self.id = agent_no

        # Make target model not trainable
        for param in target.parameters():
            param.requires_grad = False

        self.episodes = 0
        self.dru = DRU(opts['game_comm_sigma'])
        self.optimizer = optim.RMSprop(
            params=model.get_params(), lr=opts['lr'], momentum=opts['momentum'])

    def reset(self):
        """
        Resets the agent parameters
        """
        self.model.reset_parameters()
        self.model_target.reset_parameters()
        self.episodes = 0

    def _eps_flip(self, eps):
        return np.random.rand(self.opts['bs']) < eps

    def _random_choice(self, items):
        return torch.from_numpy(np.random.choice(items, 1)).item()

    def select(self, step, q, eps=0, target=False, train=False):
        """
        Returns the (action, communication) for the current step.
        """
        if not train:
            eps = 0  # Pick greedily during test

        opts = self.opts

        # Get the action range and communication range for the agent for the current time step.
        action_range, comm_range = self.game.get_action_range(step, self.id)

        action = torch.zeros(opts['bs'], dtype=torch.long)
        action_value = torch.zeros(opts['bs'])
        comm_vector = torch.zeros(opts['bs'], opts['game_comm_bits'])

        select_random_a = self._eps_flip(eps)
        for b in range(opts['bs']):
            q_a_range = range(0, opts['game_action_space'])
            a_range = range(action_range[b, 0].item() - 1, action_range[b, 1].item())
            if select_random_a[b]:
                # select action randomly (to explore the state space)
                action[b] = self._random_choice(a_range)
                action_value[b] = q[b, action[b]]
            else:
                action_value[b], action[b] = q[b, a_range].max(0)  # select action greedily
            action[b] = action[b] + 1

            q_c_range = range(opts['game_action_space'],
                                opts['game_action_space_total'])
            if comm_range[b, 1] > 0:
                # if the agent can communicate for the given time step
                c_range = range(comm_range[b, 0].item() - 1, comm_range[b, 1].item())
                # real-valued message from DRU based on q-values
                comm_vector[b] = self.dru.forward(q[b, q_c_range], train_mode=train)
        return (action, action_value), comm_vector

    def get_loss(self, episode):
        """
        Returns episodic loss for the given episodes.
        """
        opts = self.opts
        total_loss = torch.zeros(opts['bs'])
        for b in range(opts['bs']):
            b_steps = episode.steps[b].item()
            for step in range(b_steps):
                record = episode.step_records[step]
                for i in range(opts['game_nagents']):
                    td_action = 0
                    r_t = record.r_t[b][i]
                    q_a_t = record.q_a_t[b][i]

                    # Calculate td loss for environment action
                    if record.a_t[b][i].item() > 0:
                        if record.terminal[b].item() > 0:
                            td_action = r_t - q_a_t
                        else:
                            next_record = episode.step_records[step + 1]
                            q_next_max = next_record.q_a_max_t[b][i]
                            td_action = r_t = opts['gamma'] * q_next_max - q_a_t

                    loss_t = td_action ** 2
                    total_loss[b] = total_loss[b] + loss_t
        loss = total_loss.sum()
        return loss / (opts['bs'] * opts['game_nagents'])

    def update(self, episode):
        """
        Updates model parameters for given episode batch
        """
        self.optimizer.zero_grad()
        loss = self.get_loss(episode)
        loss.backward()
        # Clip gradients for stable training
        clip_grad_norm_(parameters=self.model.get_params(), max_norm=10)
        self.optimizer.step()
        self.episodes += 1

        # Update target model
        if self.episodes % self.opts['step_target'] == 0:
            self.model_target.load_state_dict(self.model.state_dict())


class Arena:
  def __init__(self, opt, game):
    self.opt = opt
    self.game = game
    self.eps = opt.eps

  def create_episode(self):
    """
    Returns an episode dictionary to maintain current episode details
    """
    opt = self.opt
    episode = DotDic({})
    episode.steps = torch.zeros(opt.bs).int()
    episode.ended = torch.zeros(opt.bs).int()
    episode.r = torch.zeros(opt.bs, opt.game_nagents).float()
    episode.step_records = []

    return episode

  def create_step_record(self):
    """
    Returns an empty step record to store the data from each step in the episode
    """
    opt = self.opt
    record = DotDic({})
    record.s_t = None
    record.r_t = torch.zeros(opt.bs, opt.game_nagents)
    record.terminal = torch.zeros(opt.bs)

    record.agent_inputs = []
    record.a_t = torch.zeros(opt.bs, opt.game_nagents, dtype=torch.long)
    record.comm = torch.zeros(opt.bs, opt.game_nagents, opt.game_comm_bits)
    record.comm_target = record.comm.clone()
    
    record.hidden = torch.zeros(opt.game_nagents, 2, opt.bs, opt.rnn_size)
    record.hidden_target = torch.zeros(opt.game_nagents, 2, opt.bs, opt.rnn_size)

    record.q_a_t = torch.zeros(opt.bs, opt.game_nagents)
    record.q_a_max_t = torch.zeros(opt.bs, opt.game_nagents)

    return record

  def run_episode(self, agents, train_mode=False):
    """
    Runs one batch of episodes for the given agents.
    """
    opt = self.opt
    game = self.game
    game.reset()
    self.eps = self.eps * opt.eps_decay

    step = 0
    episode = self.create_episode()
    s_t = game.get_state()
    # Intialize step record
    episode.step_records.append(self.create_step_record())
    episode.step_records[-1].s_t = s_t
    episode_steps = train_mode and opt.nsteps + 1 or opt.nsteps
    while step < episode_steps and episode.ended.sum() < opt.bs:
      # Run through the episode
      episode.step_records.append(self.create_step_record())

      for i in range(1, opt.game_nagents + 1):
        agent = agents[i]
        agent_idx = i - 1
        
        # Retrieve model inputs from the records
        comm = episode.step_records[step].comm.clone()
        comm_limited = self.game.get_comm_limited(step, agent.id)
        if comm_limited is not None:
          comm_lim = torch.zeros(opt.bs, 1, opt.game_comm_bits)
          for b in range(opt.bs):
            if comm_limited[b].item() > 0:
              comm_lim[b] = comm[b][comm_limited[b] - 1]
          comm = comm_lim
        else:
          comm[:, agent_idx].zero_()
        prev_action = torch.ones(opt.bs, dtype=torch.long)
        if not opt.model_dial:
          prev_message = torch.ones(opt.bs, dtype=torch.long)
        for b in range(opt.bs):
          if step > 0 and episode.step_records[step - 1].a_t[b, agent_idx] > 0:
            prev_action[b] = episode.step_records[step - 1].a_t[b, agent_idx]
        batch_agent_index = torch.zeros(opt.bs, dtype=torch.long).fill_(agent_idx)

        agent_inputs = {
          'state': episode.step_records[step].s_t[:, agent_idx],
          'messages': comm,
          'hidden': episode.step_records[step].hidden[agent_idx, :],
          'prev_action': prev_action,
          'agent': batch_agent_index
        }
        episode.step_records[step].agent_inputs.append(agent_inputs)
        
        # Get Q-values from CNet
        hidden_t, q_t = agent.model(**agent_inputs)
        episode.step_records[step + 1].hidden[agent_idx] = hidden_t.squeeze()
        # Pick actions based on q-values
        (action, action_value), comm_vector = agent.select(step, q_t, eps=self.eps, train=train_mode)

        episode.step_records[step].a_t[:, agent_idx] = action
        episode.step_records[step].q_a_t[:, agent_idx] = action_value
        episode.step_records[step + 1].comm[:, agent_idx] = comm_vector

      a_t = episode.step_records[step].a_t
      episode.step_records[step].r_t, episode.step_records[step].terminal = self.game.step(a_t)

      # Update episode record rewards
      if step < opt.nsteps:
        for b in range(opt.bs):
          if not episode.ended[b]:
            episode.steps[b] = episode.steps[b] + 1
            episode.r[b] = episode.r[b] + episode.step_records[step].r_t[b]

          if episode.step_records[step].terminal[b]:
            episode.ended[b] = 1

      # Update target network during training
      if train_mode:
        for i in range(1, opt.game_nagents + 1):
          agent_target = agents[i]
          agent_idx = i - 1

          agent_inputs = episode.step_records[step].agent_inputs[agent_idx]
          comm_target = agent_inputs.get('messages', None)

          comm_target = episode.step_records[step].comm_target.clone()
          comm_limited = self.game.get_comm_limited(step, agent.id)
          if comm_limited is not None:
            comm_lim = torch.zeros(opt.bs, 1, opt.game_comm_bits)
            for b in range(opt.bs):
              if comm_limited[b].item() > 0:
                comm_lim[b] = comm_target[b][comm_limited[b] - 1]
            comm_target = comm_lim
          else:
            comm_target[:, agent_idx].zero_()

          agent_target_inputs = copy.copy(agent_inputs)
          agent_target_inputs['messages'] = Variable(comm_target)
          agent_target_inputs['hidden'] = episode.step_records[step].hidden_target[agent_idx, :]
          hidden_target_t, q_target_t = agent_target.model_target(**agent_target_inputs)
          episode.step_records[step + 1].hidden_target[agent_idx] = hidden_target_t.squeeze()

          (action, action_value), comm_vector = agent_target.select(step, q_target_t, eps=0, target=True, train=True)

          episode.step_records[step].q_a_max_t[:, agent_idx] = action_value
          episode.step_records[step + 1].comm_target[:, agent_idx] = comm_vector

      step = step + 1
      if episode.ended.sum().item() < opt.bs:
        episode.step_records[step].s_t = self.game.get_state()

    episode.game_stats = self.game.get_stats(episode.steps)

    return episode

  def average_reward(self, episode, normalized=True):
    """
    Returns the normalized average reward for the episode.
    """
    reward = episode.r.sum()/(self.opt.bs * self.opt.game_nagents)
    if normalized:
      oracle_reward = episode.game_stats.oracle_reward.sum()/self.opt.bs
      if reward == oracle_reward:
        reward = 1
      elif oracle_reward == 0:
        reward = 0
      else:
        reward = reward/oracle_reward
    return float(reward)

  def train(self, agents, reset=True, verbose=False, test_callback=None):
    """
    Trains the agents 
    """
    opt = self.opt
    if reset:
      for agent in agents[1:]:
        agent.reset()

    self.rewards = {
        "norm_r": [],
        "step": []
    }
    for e in range(opt.nepisodes):
      episode = self.run_episode(agents, train_mode=True)
      norm_r = self.average_reward(episode)
      if verbose:
        print('train epoch:', e, 'avg steps:', episode.steps.float().mean().item(), 'avg reward:', norm_r)
      agents[1].update(episode)

      if e % opt.step_test == 0:
        episode = self.run_episode(agents, train_mode=False)
        norm_r = self.average_reward(episode)
        self.rewards['norm_r'].append(norm_r)
        self.rewards['step'].append(e)
        print('TEST EPOCH:', e, 'avg steps:', episode.steps.float().mean().item(), 'avg reward:', norm_r)


def main():
    opts = {  
        "game_nagents":3,
        "game_action_space":2,
        "game_action_space_total": 3,
        "game_comm_limited": True,
        "game_comm_bits":1,
        "game_comm_sigma":2,
        "nsteps":6,
        "gamma":1,
        "temp_min": 0.5,
        "anneal_rate": 0.0001,
        "use_gumbel": False,
        "rnn_size":128,
        "bs":32,
        "lr":0.0005,
        "momentum":0.05,
        "eps":0.05,
        "nepisodes":500,
        "step_test":10,
        "step_target":100,
        "eps_decay": 1.0
    }    
    game = SwitchGame(DotDic(opts))
    cnet = CNet(opts)
    cnet_target = copy.deepcopy(cnet)
    agents = [None]
    for i in range(1, opts['game_nagents'] + 1):
        agents.append(Agent(DotDic(opts), game=game, model=cnet, target=cnet_target, agent_no=i))

    arena = Arena(DotDic(opts), game)
    arena.train(agents)
    plt.plot('step', 'norm_r', data=arena.rewards)
    plt.xlabel('Steps')
    plt.ylabel('Normalized Reward')
    plt.title('Test Rewards')
    plt.show()

    ACTIONS = [None, 'The prisoner decided to do nothing.', 'The prisoner chose to tell.']
    game.reset()
    ep = arena.run_episode(agents, False)
    batch = 0
    game.describe_game(batch)

    for i, step in enumerate(ep.step_records[:-1]):
        print('Day', i + 1)
        active_agent = game.active_agent[batch][i].item()
        print('Prisoner Selected for the interrogation: ', active_agent)
        print(ACTIONS[step.a_t[batch].detach().numpy()[active_agent - 1]])
        if step.comm[batch].detach().numpy()[active_agent - 1][0] == 1.0:
            print('The prisoner toggled the light bulb.')
        print()
        if step.terminal[batch]:
            break

if __name__ == '__main__':
    main()
