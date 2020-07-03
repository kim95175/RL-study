import sys
import numpy as np
import tensorflow.compat.v1 as tf
import random
import gym
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

LEARNING_RATE = 0.001
GAMMA = 0.99
BUFFER_LIMIT = 5000
BATCH_SIZE = 64
INITIAL_EPSILON = 0.1
MIN_EPSILON = 0.001
EPSILON_DECAY = 0.01

class ReplayBuffer():
    def __init__(self):
        self.buffer = deque(maxlen=BUFFER_LIMIT)

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

class DQN:
    def __init__(self, env, multistep=False):

        self.env = env
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.multistep = multistep  # Multistep(n-step) 구현 시 True로 설정, 미구현 시 False
        self.n_steps = 4            # Multistep(n-step) 구현 시 n 값, 수정 가능
        self._build_network()
        self.epsilon = INITIAL_EPSILON

    def _build_network(self):
        # Target 네트워크와 Local 네트워크를 설정
        self.q = Qnet(state_size=self.state_size, action_size=self.action_size)
        self.q_target = Qnet(state_size=self.state_size, action_size=self.action_size)  # target network
        self.q_target.load_state_dict(self.q.state_dict()) # state_dict() : network의 weight 정보 포함
        self.memory = ReplayBuffer()
        self.optimizer = optim.Adam(self.q.parameters(), lr=LEARNING_RATE)
        

    def predict(self, state, epsilon):
        # state를 넣어 policy에 따라 action을 반환
        return self.q.sample_action(torch.from_numpy(state).float(), epsilon)

    def train_minibatch(self):
        # mini batch를 받아 policy를 update
        for i in range(30):
            s, a, r, next_s, done_mask = self.memory.sample(BATCH_SIZE)

            q_values = self.q(s)
            next_q_values = self.q(next_s)
            next_q_target_values = self.q_target(next_s)
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

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def update_epsilon(self):
        # Exploration 시 사용할 epsilon 값을 업데이트
        self.epsilon = self.epsilon - EPSILON_DECAY
        if self.epsilon < MIN_EPSILON:
            self.epsilon = MIN_EPSILON


    # episode 최대 횟수는 구현하는 동안 더 적게, 더 많이 돌려보아도 무방합니다.
    # 그러나 평가시에는 episode 최대 회수를 1500 으로 설정합니다.
    def learn(self, max_episode=1500):
        avg_step_count_list = []     # 결과 그래프 그리기 위해 script.py 로 반환
        last_100_episode_step_count = deque(maxlen=100)
        score = 0.0

        for episode in range(max_episode):
            epsilon = max(MIN_EPSILON, INITIAL_EPSILON - EPSILON_DECAY*(episode/200))
            done = False
            state = self.env.reset()
            step_count = 0

            # episode 시작
            while not done:
                self.update_epsilon()
                action = self.predict(state, self.epsilon)
                next_state, reward, done, _ = self.env.step(action)
                done_mask = 0.0 if done else 1.0
                self.memory.put((state, action, reward, next_state, done_mask))
                state = next_state
                step_count += 1
                if(step_count > 400):
                    self.update_epsilon()
                score += reward
                if done:
                    break
            
            if self.memory.size() > 100:
                self.train_minibatch()

            if episode % 20 == 0 and episode != 0:
                self.q_target.load_state_dict(self.q.state_dict())
                print("n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(
                                                            episode, score/20, self.memory.size(), epsilon*100))
                score = 0.0

            last_100_episode_step_count.append(step_count)
            # 최근 100개의 에피소드 평균 step 횟수를 저장 (이 부분은 수정하지 마세요)
            #if len(last_100_episode_step_count) == 100:
            avg_step_count = np.mean(last_100_episode_step_count)
            print("[Episode {:>5}]  episode step_count: {:>5} avg step_count: {}".format(episode, step_count, avg_step_count))
            avg_step_count_list.append(avg_step_count)
        
        return avg_step_count_list