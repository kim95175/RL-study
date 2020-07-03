import gym
import random
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

lr_mu = 0.0005
lr_q = 0.001
gamma = 0.99
batch_size = 2
buffer_limit = 50000
tau = 0.005

class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)
    
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
        
        return torch.tensor(s_list, dtype=torch.float), torch.tensor(a_list), \
            torch.tensor(r_list), torch.tensor(next_s_list, dtype=torch.float), \
            torch.tensor(done_mask_list)

    def size(self):
        return len(self.buffer)

# actor network
class MuNet(nn.Module):
    def __init__(self):
        super(MuNet, self).__init__()
        self.fc1 = nn.Linear(3, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc_mu = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = torch.tanh(self.fc_mu(x))*2
        return mu

# critic network
class QNet(nn.Module):
    def __init__(self):
        super(QNet, self).__init__()

        self.fc_s = nn.Linear(3, 64)
        self.fc_a = nn.Linear(1, 64)
        self.fc_q = nn.Linear(128, 32)
        self.fc_3 = nn.Linear(32, 1)

    def forward(self, x, a):
        h1 = F.relu(self.fc_s(x))
        h2 = F.relu(self.fc_a(a))
        cat = torch.cat([h1, h2], dim=1)
        q = F.relu(self.fc_q(cat))
        q = self.fc_3(q)
        return q

class OrnsteinUhlenbeckNoise:
    def __init__(self, mu):
        self.theta, self.dt, self.sigma = 0.1, 0.01, 0.1
        self.mu = mu
        self.x_prev = np.zeros_like(self.mu)

    def __call__(self):
        x= self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

def train(mu, mu_target, q, q_target, memory, q_optimizer, mu_optimzer):
    s, a, r, next_s, done_mask = memory.sample(batch_size)
    print("s", s)
    print("a", a)
    print("r", r)
    print("next_s", next_s)
    print("done", done_mask)

    # update critic by minimizing the loss
    target = r+ gamma * q_target(next_s, mu_target(next_s))
    q_loss = F.smooth_l1_loss(q(s,a), target.detach()) # minimizing the loss
    q_optimizer.zero_grad()
    q_loss.backward()
    q_optimizer.step()

    # update actor policy by the sampled policy gradient
    mu_loss = -q(s, mu(s)).mean()
    mu_optimzer.zero_grad()
    mu_loss.backward()
    mu_optimzer.step()

def soft_update(net, net_target):
    # update target_network
    for param_target, param in zip(net_target.parameters(), net.parameters()):
        param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)

def main():
    env = gym.make('Pendulum-v0')
    memory = ReplayBuffer()

    q, q_target = QNet(), QNet()
    q_target.load_state_dict(q.state_dict())
    mu, mu_target = MuNet(), MuNet()
    mu_target.load_state_dict(mu.state_dict())

    score = 0.0
    print_interval = 20

    mu_optimzer = optim.Adam(mu.parameters(), lr=lr_mu)
    q_optimizer = optim.Adam(q.parameters(), lr=lr_q)
    ou_noise = OrnsteinUhlenbeckNoise(mu=np.zeros(1))

    for n_epi in range(10000):
        s = env.reset()

        for t in range(300):
            a = mu(torch.from_numpy(s).float())
            a = a.item() + ou_noise()[0]
            next_s, r, done, _ = env.step([a])
            memory.put((s, a, r/100.0, next_s, done))
            score += r
            s = next_s

            if done:
                break
        
        if memory.size() > 2000:
            for i in range(10):
                train(mu, mu_target, q, q_target, memory, q_optimizer, mu_optimzer)
                soft_update(mu, mu_target)
                soft_update(q, q_target)
            
        if n_epi%print_interval == 0 and n_epi != 0:
            print("\r# of episode :{}, avg score : {:.1f}".format(n_epi, score/print_interval), end="")
            if n_epi%100 == 0:
                print("\r# of episode :{}, avg score : {:.1f}".format(n_epi, score/print_interval))
            score = 0.0
    
    env.close()

if __name__ == '__main__':
    main()
    