import gym
import collections
import random
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

LEARNING_RATE = 0.0005
GAMMA = 0.98
BUFFER_LIMIT = 50000
BATCH_SIZE = 64
INITIAL_EPSILON = 0.1
MIN_EPSILON = 0.01
EPSILON_DECAY = 0.01
DQN = False
DDQN = False

class SumTree():
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity -1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0

    def _propagate(self, idx, change):
        parent = (idx-1) // 2
        self.tree[parent] += change

        if parent != 0:
            self.propagate(parent, change)

    def _retrive(self, idx, s):
        left = 2 * idx + 1
        right = left +1

        if left >= len(self.tree):
            return idx
        
        if s <=self.tree[left]:
            return self._retrive(left, s)
        else:
            return self._retrive(right, s - self.tree[left])
    
    def total(self):
        return self.tree[0]

    def add(self, p , data):
        idx = self.write + self.capacity-1
        self.data[self.write] = data
        self.update(idx, p)

        self.write +=1
        if self.write >= self.capacity:
            self.write = 0
        
        if self.n_entries < self.capacity:
            self.n_entries += 1
    
    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)
    
    def get(self, s):
        idx = self._retrive(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])

class PERBuffer():
    e = 0.01
    a = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001

    def __init__(self, capacity):
        self.tree = SumTree(capacity=capacity)
        self.capacity = capacity

    def _get_priority(self, error):
        return (np.abs(error) + self.e) ** self.a
    
    def add(self, error, sample):
        p = self._get_priority(error)
        self.tree.add(p, sample)
    
    def sample(self, n):
        batch = []
        idxs = []
        segment = self.tree.total()/ n 
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(n):
            a = segment * i
            b = segment * (i+1)
            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)
        
        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        return batch, idx, is_weight
    
    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)
        
class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=BUFFER_LIMIT)

    def put(self, transition):
        self.buffer.append(transition)
    
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_list, a_list, r_list, next_s_list, done_mask_list = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, next_s, done_mask = transition
            s_list.append(s)
            a_list.append([a])
            r_list.append([r])
            next_s_list.append(next_s)
            done_mask_list.append([done_mask])
        
        return torch.tensor(s_list, dtype=torch.float), \
            torch.tensor(a_list), \
            torch.tensor(r_list), \
            torch.tensor(next_s_list, dtype=torch.float), \
            torch.tensor(done_mask_list)
    
    def size(self):
        return len(self.buffer)

class Qnet(nn.Module):
    def __init__(self, state_size, action_size):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def sample_action(self, obs, epsilon):
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0,1)
        else:
            return out.argmax().item()
    
def train(q, q_target, memory, optimzer):
    for i in range(10):
        if DDQN == True:
            s, a, r, next_s, done_mask = memory.sample(BATCH_SIZE)
            q_values = q(s)
            next_q_values = q(next_s)
            next_q_target_values = q_target(next_s)

            q_value = q_values.gather(1, a)
            next_q_value = next_q_target_values.gather(1, next_q_values.max(1)[1].unsqueeze(1))
            target = r + GAMMA * next_q_value * done_mask
            loss = F.smooth_l1_loss(q_value, target)

            optimzer.zero_grad()
            loss.backward()
            optimzer.step()
        else:
            s, a, r, next_s, done_mask = memory.sample(BATCH_SIZE)
            q_out = q(s)
            q_a = q_out.gather(1, a)
            max_q_prime = q_target(next_s).max(1)[0].unsqueeze(1) 
            target = r + GAMMA * max_q_prime * done_mask
            loss = F.smooth_l1_loss(q_a, target)
            optimzer.zero_grad()
            loss.backward()
            optimzer.step()

def PER_train(q, q_target, memory, optimizer, epsilon):
    mini_batch, idxs, is_weights = memory.sample(BATCH_SIZE)
    mini_batch = np.array(mini_batch).transpose()

    states = np.vstack(mini_batch[0])
    actions = list(mini_batch[1])
    rewards = list(mini_batch[2])
    next_states = np.vstack(mini_batch[3])
    dones = mini_batch[4]

    states = torch.Tensor(states)
    states = torch.autograd.Variable(states).float()
    pred = q(states)

    a = torch.LongTensor(actions).view(-1, 1)
    one_hot_action = torch.FloatTensor(BATCH_SIZE, q.action_size).zero()
    one_hot_action.scatter_(1, a, 1)
    
    pred = torch.sum(pred.mul(torch.autograd.Variable(one_hot_action)), dim=1)

    next_states = torch.Tensor(next_states)
    next_states = torch.autograd.Variable(next_states).float()
    next_pred = q_target(next_states).data

    rewards = torch.FloatTensor(rewards)
    dones = torch.FloatTensor(dones)

    target = rewards + dones * GAMMA * next_pred.max(1)[0]
    target = torch.autograd.Variable(target)

    errors = torch.abs(pred-target).data.numpy()

    for i in range(BATCH_SIZE):
        idx = idxs[i]
        memory.update(idx, errors[i])
    
    optimizer.zero_grad()
    
    loss = (torch.FloatTensor(is_weights) * F.mse_loss(pred, target)).mean()
    loss.backward()

    optimizer.step()

def main():
    parser = argparse.ArgumentParser(description='policy')
    parser.add_argument('--env', default='MountainCar-v0',
                        help='Name of the environment (default: %(default)s)')
    parser.add_argument('--policy', default ='dqn',
                        help='Name of policy')
    #parser.add_argument('--episodes', type=int, default=1,help='episodes (default: %(default)s)')
    args = parser.parse_args()
    print("=" * 70)
    print("env : {}    policy : {}".format(args.env, args.policy))
    print("=" * 70)
    env = gym.make(args.env)
    policy = args.policy
    if policy == 'ddqn':
        DDQN = True
    else:
        DQN = True

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    print("observation_space : {}/{}    action_space : {}/{}".format(
                    env.observation_space, state_size, env.action_space, action_size))
    print("=" * 70)
    q = Qnet(state_size=state_size, action_size=action_size)
    q_target = Qnet(state_size=state_size, action_size=action_size)  # target network
    q_target.load_state_dict(q.state_dict()) # state_dict() : network의 weight 정보 포함
    memory = ReplayBuffer()

    print_interval = 10
    score = 0.0
    optimizer = optim.Adam(q.parameters(), lr=LEARNING_RATE)
    
    for n_epi in range(10000):
        epsilon = max(MIN_EPSILON, INITIAL_EPSILON - EPSILON_DECAY*(n_epi/200))
        s = env.reset()
        done = False
    
        while not done:
            #if n_epi > 5000 and n_epi % 100 == 0:
            #    env.render()
            a = q.sample_action(torch.from_numpy(s).float(), epsilon)                
            next_s, r, done, _ = env.step(a)
            done_mask = 0.0 if done else 1.0
    
            memory.put((s,a,r/100.0, next_s, done_mask))
            s = next_s
            score += r

            if done:
                   break 
            
        if memory.size() > 2000:
            train(q, q_target, memory, optimizer)
            
        if n_epi % print_interval==0 and n_epi != 0:
            q_target.load_state_dict(q.state_dict())
            print("n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(
                                                                n_epi, score/print_interval, memory.size(), epsilon*100))
            score = 0.0
        env.close()

if __name__ == '__main__':
    main()

        