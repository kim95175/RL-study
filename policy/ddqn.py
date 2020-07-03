import gym
import collections
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

LEARNING_RATE = 0.0005
GAMMA = 0.99
BUFFER_LIMIT = 50000
BATCH_SIZE = 64
INITIAL_EPSILON = 0.1
MIN_EPSILON = 0.01
EPSILON_DECAY = 0.02

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
        s, a, r, next_s, done_mask = memory.sample(BATCH_SIZE)

        #q_out = q(s)
        #q_a = q_out.gather(1, a)
        #max_q_prime = q_target(next_s).max(1)[0].unsqueeze(1) 
        q_values = q(s)
        next_q_values = q(next_s)
        next_q_target_values = q_target(next_s)

        q_value = q_values.gather(1, a)
        '''
        print(q_values.shape)
        print(a.shape)
        print(next_q_target_values.shape)
        print(next_q_values.max(1)[1].unsqueeze(1).shape)
        '''
        next_q_value = next_q_target_values.gather(1, next_q_values.max(1)[1].unsqueeze(1))
        target = r + GAMMA * next_q_value * done_mask
        loss = F.smooth_l1_loss(q_value, target)

        optimzer.zero_grad()
        loss.backward()
        optimzer.step()

def main():
    #env = gym.make('MountainCar-v0')
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    print(env.observation_space.shape[0])
    print(env.action_space.n)
    q = Qnet(state_size=state_size, action_size=action_size)
    q_target = Qnet(state_size=state_size, action_size=action_size)  # target network
    q_target.load_state_dict(q.state_dict()) # state_dict() : network의 weight 정보 포함
    memory = ReplayBuffer()

    print_interval = 20
    score = 0.0
    optimizer = optim.Adam(q.parameters(), lr=LEARNING_RATE)

    for n_epi in range(1500):
        epsilon = max(MIN_EPSILON, INITIAL_EPSILON - EPSILON_DECAY*(n_epi/200))
        s = env.reset()
        done = False
        
        while not done:
            if n_epi > 9000 and n_epi % 100 == 0:
                env.render()
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

        