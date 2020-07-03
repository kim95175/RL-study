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

        self.double_q = double_q    # Double DQN        구현 시 True로 설정, 미구현 시 False
        self.per = per              # PER               구현 시 True로 설정, 미구현 시 False
        self.multistep = multistep  # Multistep(n-step) 구현 시 True로 설정, 미구현 시 False

        self.n_steps = 1            # Multistep(n-step) 구현 시의 n 값
        self.eps = 1.0
        self.batch_size = 8
        self.memory = deque(maxlen=REPLAY_MEMORY)

        self.local_network = self._build_network()
        self.target_network = self._build_network()
        self.target_network.set_weights(self.local_network.get_weights())
        
        self.memory_counter = 0
        self.learning_step = 0
        self.target_replace_iter = 100

    def _build_network(self):
        # Local 네트워크를 설정
        model = models.Sequential()
        model.add(layers.Dense(24, activation='relu', input_dim= self.state_size))
        model.add(layers.Dense(24, activation='relu'))
        model.add(layers.Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=LEARNING_RATE))
        return model
        '''
        self._sess = sess
        with tf.variable_scope('q'):
            self._x = tf.placeholder(
                tf.float32, shape=[None, self.state_size], name='input_x'
            )
            net = self._x
            net = tf.layers.dense(net, units=self.hidden_size, activation=tf.nn.relu, name="dense/1")
            net = tf.layers.dense(net, units=self.action_size, name="dense/2")
            self._q_pred = net

            self._y = tf.placeholder(
                    tf.float32, shape=[None, self.action_size], name='input_y'
                )
            #self._loss = tf.reduce_sum(tf.square(self._y - self._q_pred))
            self._loss = tf.losses.mean_squared_error(self._y, self._q_pred)
            optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)
            self._train = optimizer.minimize(self._loss)

        with tf.variable_scope('target_q'):
            self.target_x = tf.placeholder(
                tf.float32, shape=[None, self.state_size], name='target_input_x'
            )
            target_net = self.target_x
            target_net = tf.layers.dense(target_net, units=self.hidden_size, activation=tf.nn.relu, name="t_dense/1")
            target_net = tf.layers.dense(target_net, units=self.action_size, name="t_dense/2")
            self.target_q_pred = target_net
        '''

    def predict(self, state):
        # state를 넣어 policy에 따라 action을 반환
        if(np.random.rand() < self.eps) :
            #print("random")
            return self.env.action_space.sample()
        else:
            #cur_state = np.reshape(state, [1, self.state_size])
            #action_value = self._sess.run(self._q_pred, feed_dict={self._x: cur_state})[0]
            #print(action_value)
            #return np.argmax(action_value)
            print(self.local_network.predict(state))
            return np.argmax(self.local_network.predict(state))

    def train_minibatch(self, mini_batch):
        # mini batch를 받아 policy를 update
        
        samples = np.array(mini_batch)
        state, action, reward, next_state, done = np.hsplit(samples, 5)
        #states = np.concatenate((state[:]), axis = 0)
        states = np.concatenate((np.squeeze(state[:])), axis = 0)
        states = np.reshape(states, (self.batch_size, 2))
        rewards = reward.reshape(self.batch_size,).astype(float)
        targets = self.local_network.predict(states)
        next_states = np.concatenate((np.squeeze(next_state[:])), axis = 0)
        next_states = np.reshape(next_states, (self.batch_size, 2))
        dones = np.concatenate(done).astype(bool)
        notdones = (dones^1).astype(float)
        dones = dones.astype(float)
        Q_futures = self.target_network.predict(next_states).max(axis = 1)
        targets[(np.arange(self.batch_size), action.reshape(self.batch_size,).astype(int))] = rewards * dones + (rewards + Q_futures * DISCOUNT_RATE)*notdones
        self.local_network.fit(states, targets, epochs=1, verbose=0)
        '''
        states = np.concatenate((np.squeeze(state[:])), axis = 0)
        states = np.reshape(states, (self.batch_size, 2))
        next_states = np.concatenate((np.squeeze(next_state[:])), axis = 0)
        next_states = np.reshape(next_states, (self.batch_size, 2))
        dones = np.concatenate(done).astype(int)
        dones = np.reshape(dones, (self.batch_size))
        notdones = (dones^1).astype(float)
        rewards = reward.reshape(self.batch_size).astype(float)
        targets = self._sess.run(self._q_pred, feed_dict={self._x: states})
        target_Q = self._sess.run(self._q_pred, feed_dict={self._x: next_states})
        maxQ = target_Q.max(axis=1)
        targets[(np.arange(self.batch_size), action.reshape(self.batch_size,).astype(int))] = (rewards * dones) + ((rewards + (maxQ * DISCOUNT_RATE)) * notdones )
        loss, train = self._sess.run([self._loss, self._train], feed_dict= {self._x:states, self._y:targets})
        '''
    
        '''
        state_minibatch = []
        y_minibatch = []
        for state, action, reward, next_state, done in mini_batch:
            cur_state = np.reshape(state, [1, self.state_size])
            Q = self._sess.run(self._q_pred, feed_dict={self._x: cur_state})[0]

            if done:
                Q[action] = reward
            else:
                n_state = np.reshape(next_state, [1, self.state_size])
                Q_value = self._sess.run(self._q_pred, feed_dict={self._x: n_state})[0]
                Q[action] = reward + DISCOUNT_RATE * np.max(Q_value)
            
            state_minibatch.append(state)
            y_minibatch.append(Q)
        
        loss, train = self._sess.run([self._loss, self._train], feed_dict= {self._x:state_minibatch, self._y:y_minibatch})
        
        return loss, train
        '''

    def update_epsilon(self, num_episode) :
        # Exploration 시 사용할 epsilon 값을 업데이트
        tmp_eps = 1.0 / ( (num_episode//500)+1)
        self.eps = max(0.001,  tmp_eps)
        

    def target_update(self):
        op_list = []
        src_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='q')
        dest_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target_q')    
        for src_var, dest_var in zip(src_vars, dest_vars):
            op_list.append(dest_var.assign(src_var.value()))
        return op_list

    # episode 최대 회수는 구현하는 동안 더 적게, 더 많이 돌려보아도 무방합니다.
    # 그러나 제출 시에는 최종 결과 그래프를 그릴 때는 episode 최대 회수를
    # 1000 번으로 고정해주세요. (다른 학생들과의 성능/결과 비교를 위해)
    def learn(self, max_episode:int = 1000):
        episode_record = []     # plot을 그리기 위해 데이터를 저장하는 list
        last_100_game_reward = deque(maxlen=100)
        max_episode = 10000
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
            # episode 시작
            while not done:
                action = self.predict(state)
                next_state, reward, done, _ = self.env.step(action)
                if done and step_count < 199 :
                    reward = 10
                if next_state[0] > - 0.4:
                    reward = (1+next_state[0])**2

                if step_count % render_count == 0:
                    self.env.render()
                #print((state, action, reward, next_state, done))            
                self.memory.append((state, action, reward, next_state, done))
                episode_reward += reward
                state = next_state
                step_count += 1

            if episode % 1 == 0:
                if len(self.memory) > self.batch_size :
                    for _ in range(50):
                        mini_batch = random.sample(self.memory, self.batch_size)
                        self.train_minibatch(mini_batch)
                        #print("Loss: ", loss)
                        #sess.run(self.target_update())
                        self.target_network.set_weights(self.local_network.get_weights())
        
        
                # 최근 100개의 에피소드 reward 평균을 저장
                last_100_game_reward.append(episode_reward)
                avg_reward = np.mean(last_100_game_reward)
                if avg_reward > -100 : 
                    render_count = 25
                if avg_reward > -50:
                    render_count = 10
                episode_record.append(avg_reward)
                if step_count >= 199:
                    print("[Episode Failed {:>5}]  episode steps: {:>5} avg: {}".format(episode, step_count, avg_reward))
                else:
                    print("[Episode Success {:>5}]  episode steps: {:>5} avg: {}".format(episode, step_count, avg_reward))
                    #print("[Episode {:>5}]  episode steps: {:>5} avg: {}".format(episode, step_count, avg_reward))
        '''
        exploration_step = 200
        num_for_ready = 0
        while exploration_step >= 199:
            num_for_ready += 1
            exploration_step = 0
            state = self.env.reset()
            done_ = False
            tmp_memory = []
            while not done_:
                action = self.env.action_space.sample()
                next_state, reward, done_, _ = self.env.step(action)
                if exploration_step % 50 == 0:
                    self.env.render()
                tmp_memory.append((state, action, reward, next_state, done_))
                exploration_step += 1
                state = next_state  

        print("Flag done in ",num_for_ready)
        print("Let's START!!!!")
        
        for i in range(len(tmp_memory)):
             self.memory.append(tmp_memory[i])

        with tf.Session() as sess:
            #sess.run(self.target_update())
            self._build_network(sess)
            sess.run(tf.global_variables_initializer()) 
        '''
        return episode_record

