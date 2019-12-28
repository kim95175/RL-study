import sys
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow.compat.v1.keras as keras
from keras import models
from keras import layers
import math
import random
import gym
from collections import deque

DISCOUNT_RATE = 0.99            # gamma parameter
REPLAY_MEMORY = 50000           # Replay buffer 의 최대 크기
LEARNING_RATE = 0.001           # learning rate parameter default = 0.001
LEARNING_STARTS = 1000          # 1000 스텝 이후 training 시작


class DQN:
    def __init__(self, env, double_q=False, multistep=False, per=False):

        self.env = env
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.hidden_size = 32

        self.double_q = double_q  # Double DQN        구현 시 True로 설정, 미구현 시 False
        self.per = per              # PER               구현 시 True로 설정, 미구현 시 False
        self.multistep = multistep  # Multistep(n-step) 구현 시 True로 설정, 미구현 시 False

        self.n_steps = 3            # Multistep(n-step) 구현 시의 n 값
        self.eps = 0
        self.batch_size = 128
        self.memory = deque(maxlen=REPLAY_MEMORY)

        self.local_network = self._build_network()
        self.target_network = self._build_network()
        self.target_network.set_weights(self.local_network.get_weights())
        
        self.memory_counter = 0
        self.learning_step = 20
        self.target_replace_iter = 100

    def _build_network(self):
        # Local 네트워크 및 target 네트워크를 설정
        model = models.Sequential()
        model.add(layers.Dense(self.hidden_size, activation='relu', input_dim= (self.state_size) ))
        model.add(layers.Dense(self.hidden_size, activation='relu'))
        model.add(layers.Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=keras.optimizers.RMSprop(lr=LEARNING_RATE))
        return model


    def predict(self, state):
        # state를 넣어 policy에 따라 action을 반환
        if(np.random.rand() < self.eps) :
            return self.env.action_space.sample()
        else:
            state = state.reshape(1,2)
            return np.argmax(self.local_network.predict(state))

    def train_minibatch(self, mini_batch):
        # mini batch를 받아 policy를 update
        samples = np.array(mini_batch)
        state, action, reward, next_state, done = np.hsplit(samples, 5)
        sampe_states = np.concatenate((np.squeeze(state[:])), axis = 0)
        sampe_states = np.reshape(sampe_states, (self.batch_size, 2))
        sample_rewards = reward.reshape(self.batch_size,).astype(float)
        Q = self.local_network.predict(sampe_states)
        sampe_nstates = np.concatenate((np.squeeze(next_state[:])), axis = 0)
        sampe_nstates = np.reshape(sampe_nstates, (self.batch_size, 2))
        dones = np.concatenate(done).astype(bool)
        not_dones = (dones^1).astype(float)
        dones = dones.astype(float)
        if self.double_q == True:
            target_next_q = self.target_network.predict(sampe_nstates)
            double_action = np.argmax(self.local_network.predict(sampe_nstates), axis=1)
            next_q = target_next_q[(np.arange(self.batch_size), double_action)]
            Q[(np.arange(self.batch_size), action.reshape(self.batch_size,).astype(int))] = sample_rewards * dones + (sample_rewards + next_q * DISCOUNT_RATE)*not_dones
        else:
            next_q = self.target_network.predict(sampe_nstates).max(axis = 1)
            Q[(np.arange(self.batch_size), action.reshape(self.batch_size,).astype(int))] = sample_rewards * dones + (sample_rewards + next_q * DISCOUNT_RATE)*not_dones
        self.local_network.fit(sampe_states, Q, epochs=1, verbose=0)
       

    def update_epsilon(self, num_episode) :
        # Exploration 시 사용할 epsilon 값을 업데이트
        tmp_eps = 1.0 / ((num_episode//50) + 1)
        self.eps = max(0.001,  tmp_eps)
        
    # episode 최대 회수는 구현하는 동안 더 적게, 더 많이 돌려보아도 무방합니다.
    # 그러나 제출 시에는 최종 결과 그래프를 그릴 때는 episode 최대 회수를
    # 1000 번으로 고정해주세요. (다른 학생들과의 성능/결과 비교를 위해)
    def learn(self, max_episode:int = 1000):
        episode_record = []     # plot을 그리기 위해 데이터를 저장하는 list
        last_100_game_reward = deque(maxlen=100)
        render_count = 50
        print("=" * 70)
        print("Double : {}    Multistep : {}/{}    PER : {}".format(self.double_q, self.multistep, self.n_steps, self.per))
        print("=" * 70)

        for episode in range(max_episode):
            done = False
            state = self.env.reset()
            step_count = 0
            episode_reward = 0
            self.update_epsilon(episode)
            tmp_reward= 0 
            multi_state = []
            multi_action= []
            multi_reward= []
            multi_next_state = []
            multi_done = []
            # episode 시작
            while not done:
                action = self.predict(state)
                next_state, reward, done, _ = self.env.step(action)
                tmp_reward = reward
                if next_state[0] >= 0.5:
                    tmp_reward = 10
                if next_state[0] > -0.4 :
                    tmp_reward = -0.1 + next_state[0]
                if step_count % render_count == 0:
                    self.env.render()
                
                if self.multistep == True:
                    nstep = self.n_steps
                    len_tmpmem = len(multi_state)
                    if done:
                        for i in range(len_tmpmem):
                            if i == (len_tmpmem-nstep):
                                nstep -= 1
                            for j in range(nstep-1):
                                multi_reward[i] += multi_reward[i+j+1] * (DISCOUNT_RATE ** (j+1))
                            multi_next_state[i] = multi_state[i+nstep]
                            self.memory.append((multi_state[i], multi_action[i], multi_reward[i], multi_next_state[i], multi_done[i]))
                    else :
                        multi_state.append(state)
                        multi_action .append(action)
                        multi_reward.append(tmp_reward)
                        multi_next_state.append(next_state)
                        multi_done.append(done)       
                else:
                    self.memory.append((state, action, tmp_reward, next_state, done))
                episode_reward += reward
                state = next_state
                step_count += 1

            if episode % 1 == 0:
                if len(self.memory) > LEARNING_STARTS:
                    for _ in range(self.learning_step):
                        mini_batch = random.sample(self.memory, self.batch_size)
                        self.train_minibatch(mini_batch)
                        self.memory_counter +=1
                if self.memory_counter >= self.target_replace_iter:
                    self.memory_counter = 0
                    self.target_network.set_weights(self.local_network.get_weights())
                
                # 최근 100개의 에피소드 reward 평균을 저장
            last_100_game_reward.append(episode_reward)
            avg_reward = np.mean(last_100_game_reward)
            episode_record.append(avg_reward)
            if step_count >= 199:
                if episode % 10 == 0:
                    print("[Failed {:>5}] Reward {:.5f}  episode steps: {:>4} avg: {}".format(episode, episode_reward, step_count, avg_reward))
            else:
                print("[Success {:>5}] Reward {:.5f} episode steps: {:>4} avg: {}".format(episode, episode_reward, step_count, avg_reward))
                #print("[Episode {:>5}]  episode steps: {:>5} avg: {}".format(episode, step_count, avg_reward))

        return episode_record

