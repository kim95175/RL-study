import argparse
import gym
from ma_gym.wrappers import Monitor
import numpy as np

import collections
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

LEARNING_RATE = 0.0005
GAMMA = 0.99
BUFFER_LIMIT = 50000
BATCH_SIZE = 128
INITIAL_EPSILON = 0.2
EPSILON_DECAY = 0.01
MIN_EPSILON = 0.02


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
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
    
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

        q_out = q(s)
        #print("q_out : ", q_out.shape)
        #print(q_out)
        q_a = q_out.gather(1, a)
        #print("q_a : ", q_a.shape)
        #print(q_a)
        max_q_prime = q_target(next_s).max(1)[0].unsqueeze(1)
        #print("max_q_prime : ", max_q_prime.shape)
        #print(max_q_prime)
        target = r + GAMMA * max_q_prime * done_mask
        #print("target : ", target.shape)
        #print(target)
        loss = F.smooth_l1_loss(q_a, target)

        optimzer.zero_grad()
        loss.backward()
        optimzer.step()


def seperate_network():
    print("train using seperate network")
    parser = argparse.ArgumentParser(description='Random Agent for ma-gym')
    parser.add_argument('--env', default='Checkers-v0',
                        help='Name of the environment (default: %(default)s)')
    #parser.add_argument('--episodes', type=int, default=1,help='episodes (default: %(default)s)')
    args = parser.parse_args()
    env = gym.make(args.env)
    print("env = ", args.env)
    #env = Monitor(env, directory='recordings/' + args.env, force=True)
    n_agent = env.n_agents
    state_size = env.observation_space[0].shape[0]
    action_size = env.action_space[0].n
    print(state_size)
    print(action_size)
    multi_q = [Qnet(state_size=state_size, action_size=action_size) for _ in range(n_agent)]
    multi_q_target = [Qnet(state_size=state_size, action_size=action_size) for _ in range(n_agent)] # target network
    for i in range(n_agent):
        multi_q_target[i].load_state_dict(multi_q[i].state_dict()) # state_dict() : network의 weight 정보 포함

    multi_memory = [ReplayBuffer() for _ in range(n_agent)]
    print_interval = 10
    update_interval = 10
    render_interval = 100
    train_interval = 1
    score = [0.0 for _ in range(n_agent)]
    multi_optimizer = [optim.Adam(multi_q[i].parameters(), lr=LEARNING_RATE) for i in range(n_agent)]
    step_counts=0

    for ep_i in range(10000):
        done_n = [False for _ in range(n_agent)]
        done_mask_n = [1.0 for _ in range(n_agent)]
        ep_reward = 0
        epsilon = max(MIN_EPSILON, INITIAL_EPSILON - EPSILON_DECAY*(ep_i/200))
        env.seed(ep_i)
        s_n = env.reset()
        #print("s_n \n", s_n)
        action_n = env.action_space.sample()
        food_collect= {0: {'lemon': 0, 'apple': 0},
                             1: {'lemon': 0, 'apple': 0}}
        
        while not all(done_n):
            for i in range(n_agent):
                #print("{}'s state : {}".format(i, s_n[i]))
                tmp_s = np.array(s_n[i])
                action_n[i] = multi_q[i].sample_action(torch.from_numpy(tmp_s).float(), epsilon)
            next_s_n, reward_n, done_n, info = env.step(action_n)
            common_reward = reward_n[0] + reward_n[1]
            #print(reward_n)
            for i in range(n_agent):
                
                for food in ['lemon', 'apple']:
                    if reward_n[i] == env.agent_reward[i][food] - 0.01:
                        food_collect[i][food] += 1
                done_mask_n[i] = 0.0 if done_n[i] else 1.0
                #multi_memory[i].put((s_n[i], action_n[i], reward_n[i], next_s_n[i], done_mask_n[i]))
                multi_memory[i].put((s_n[i], action_n[i], common_reward, next_s_n[i], done_mask_n[i]))
            s_n = next_s_n
            #ep_reward += sum(reward_n)
            for i in range(n_agent):
                score[i] += reward_n[i]
            if ep_i > 9000 and ep_i % render_interval==0 :
                env.render()
    
        step_counts += env._step_count
        if multi_memory[0].size() > 5000 and ep_i % train_interval == 0:
            for i in range(n_agent):
                train(multi_q[i], multi_q_target[i], multi_memory[i], multi_optimizer[i])
        
        if ep_i % update_interval ==0 and ep_i != 0:
            for i in range(n_agent):
                multi_q_target[i].load_state_dict(multi_q[i].state_dict()) 

        if ep_i % print_interval==0 and ep_i != 0:
            total_score = (score[0] + score[1]) / print_interval
            step_count = step_counts / print_interval
        
            print("\rn_episode :{:>4.0f}, score : {:>4.1f} [{:>4.1f}/{:>4.1f}], step_count : {:3.0f}, eps : {:.1f}% {} {}".format(
                            ep_i, total_score, score[0]/print_interval, score[1]/print_interval, 
                            step_count, epsilon*100, list(food_collect[0].values()), list(food_collect[1].values())), end = "")
            if ep_i% 100 == 0 or ( food_collect[0] == [9, 0] ):
                print("\rn_episode :{:>4.0f}, score : {:>4.1f} [{:>4.1f}/{:>4.1f}], step_count : {:3.0f}, eps : {:.1f}% {} {}".format(
                            ep_i, total_score, score[0]/print_interval, score[1]/print_interval, 
                            step_count, epsilon*100, list(food_collect[0].values()), list(food_collect[1].values())))
            '''
            print("\rn_episode :{:>4.0f}, score : {:>4.1f} [{:>4.1f}/{:>4.1f}], step_count : {:3.0f}, eps : {:.1f}%\t".format(
                            ep_i, total_score, score[0]/print_interval, score[1]/print_interval, step_count, epsilon*100), end = "")
            if ep_i% 100 == 0:
                print("\rn_episode :{:>4.0f}, score : {:>4.1f} [{:>4.1f}/{:>4.1f}], step_count : {:3.0f}, eps : {:.1f}%\t".format(
                            ep_i, total_score, score[0]/print_interval, score[1]/print_interval, step_count, epsilon*100))
            '''
            score = [0.0 for _ in range(n_agent)]
            #ep_reward = 0
            step_counts = 0
        env.close()

        #print('\rEpisode #{} Reward: {}'.format(ep_i, ep_reward), end="")
    #env.close()

def one_network():
    print("train using one network")
    parser = argparse.ArgumentParser(description='Random Agent for ma-gym')
    parser.add_argument('--env', default='Checkers-v1',
                        help='Name of the environment (default: %(default)s)')
    #parser.add_argument('--episodes', type=int, default=1,help='episodes (default: %(default)s)')
    args = parser.parse_args()
    env = gym.make(args.env)
    #env = Monitor(env, directory='recordings/' + args.env, force=True)
    n_agent = env.n_agents
    state_size = env.observation_space[0].shape[0]
    action_size = env.action_space[0].n
    print(state_size)
    print(action_size)
    q = Qnet(state_size=state_size, action_size=action_size)
    q_target = Qnet(state_size=state_size, action_size=action_size) # target network
    q_target.load_state_dict(q.state_dict()) # state_dict() : network의 weight 정보 포함

    memory = ReplayBuffer()
    print_interval = 10
    update_interval = 10
    score = [0.0 for _ in range(n_agent)]
    optimizer = optim.Adam(q.parameters(), lr=LEARNING_RATE)


    for ep_i in range(10000):
        done_n = [False for _ in range(n_agent)]
        done_mask_n = [1.0 for _ in range(n_agent)]
        ep_reward = 0
        epsilon = max(MIN_EPSILON, INITIAL_EPSILON - EPSILON_DECAY*(ep_i/200))
        env.seed(ep_i)
        s_n = env.reset()
        #print("s_n \n", s_n)
        action_n = env.action_space.sample()
        while not all(done_n):
            #env.render()
            for i in range(n_agent):
                #print("{}'s state : {}".format(i, s_n[i]))
                tmp_s = np.array(s_n[i])
                action_n[i] = q.sample_action(torch.from_numpy(tmp_s).float(), epsilon)
            next_s_n, reward_n, done_n, info = env.step(action_n)
            #print(done_n)
            for i in range(n_agent):
                done_mask_n[i] = 0.0 if done_n[i] else 1.0
                memory.put((s_n[i], action_n[i], reward_n[i], next_s_n[i], done_mask_n[i]))
            #print("next_s \n", next_s_n)
            #print("reward_n \n", reward_n)
            #print("done_n \n", done_n)
            ep_reward += sum(reward_n)
            for i in range(n_agent):
                score[i] += reward_n[i]
            s_n = next_s_n
            if ep_i > 5000 and ep_i % print_interval==0 :
                env.render()
            
        if memory.size() > 5000:
            train(q, q_target, memory, optimizer)
        
        if ep_i % update_interval ==0 and ep_i != 0:
            q_target.load_state_dict(q.state_dict()) 

        if ep_i % print_interval==0 and ep_i != 0:
            #for i in range(n_agent):
            total_score = (score[0] + score[1]) / print_interval
            print("\rn_episode :{:>4.0f}, score : {:>4.1f} [{:>4.1f}/{:>4.1f}], n_buffer : {}, eps : {:.1f}%\t".format(
                            ep_i, total_score, score[0]/print_interval, score[1]/print_interval, memory.size(), epsilon*100), end = "")
            if ep_i% 100 == 0:
                print("\rn_episode :{:>4.0f}, score : {:>4.1f} [{:>4.1f}/{:>4.1f}], n_buffer : {}, eps : {:.1f}%\t".format(
                            ep_i, total_score, score[0]/print_interval, score[1]/print_interval, memory.size(), epsilon*100))
            score = [0.0 for _ in range(n_agent)]
        env.close()

        #print('Episode #{} Reward: {}'.format(ep_i, ep_reward))
    #env.close()


if __name__ == '__main__':
    #one_network()
    seperate_network()
    