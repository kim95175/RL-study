import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.layers as kl
import tensorflow.keras.optimizers as ko
import tensorflow.keras as k
from tensorflow.keras.initializers import RandomUniform as RU
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import gym

class _actor_network():
    def __init__(self, state_dim, action_dim, action_bound_range=1):
        self.state_dim= state_dim
        self.action_dim = action_dim
        self.action_bound_range = action_bound_range

    def model(self):
        state = kl.Input(shape=self.state_dim, dtype='float32')
        x = kl.Dense(400, activation='relu', kernel_initializer=RU(-1/np.sqrt(self.state_dim) , 1/np.sqrt(self.state_dim)))(state)
        x = kl.Dense(300, activation='relu', kernel_initializer=RU(-1/np.sqrt(400), 1/np.sqrt(400)))(x)
        out = kl.Dense(self.action_dim, activation='tanh', kernel_initializer=RU(-0.003, 0.003))(x)
        return k.Model(inputs=state, outputs=out)

class _critic_network():
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
    
    def model(self):
        state = kl.Input(shape=self.state_dim, name='state_input', dtype='float32')
        state_i = kl.Dense(400, activation='relu', 
                    kernel_initializer=RU(-1/np.sqrt(self.state_dim) , 1/np.sqrt(self.state_dim)))(state)
        action = kl.Input(shape=(self.action_dim,), name='action_input')
        x = kl.concatenate([state_i, action])
        x = kl.Dense(300, activation='relu',
                kernel_initializer=RU(-1/np.sqrt(401) , 1/np.sqrt(401)))(x)
        out = kl.Dense(1, activation='linear')(x) 
        return k.Model(inputs=[state, action], outputs=out)
        
class Replay_Buffer:
    def __init__(self, max_buffer_size, batch_size, dflt_dtype='float32'):
        self.batch_size = batch_size
        self.buffer = deque(maxlen=max_buffer_size)
        self.dflt_dtype = dflt_dtype

    def store(self, s, a, r, n_s, done):
        self.buffer.append([s, a, r, n_s, done])
    
    def sample_batch(self):
        replay_buffer = np.array(random.sample(self.buffer, self.batch_size))
        arr = np.array(replay_buffer)
        s_batch = np.vstack(arr[:, 0])
        a_batch = arr[:, 1].astype(self.dflt_dtype).reshape(-1, 1)
        r_batch = arr[:, 2].astype(self.dflt_dtype).reshape(-1, 1)
        n_s_batch = np.vstack(arr[:, 3])
        done_batch = np.vstack(arr[:, 4]).astype(bool)
        return s_batch, a_batch, r_batch, n_s_batch, done_batch

class DDPG():
    def __init__(self, env,
                max_buffer_size= int(1e5),
                batch_size = 32,
                max_time_stpes = 1000,
                tau = 0.005,
                gamma = 0.99,
                explore_time =1000,
                actor_lr = 0.001,
                critic_lr = 0.001,
                dtype = 'float32',
                n_episodes = 200,
                ):
        self.env = env
        self.max_buffer_size = max_buffer_size
        self.batch_size = batch_size
        self.T = max_time_stpes
        self.tau = tau
        self.gamma = gamma
        self.explore_time = explore_time
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.n_episodes = n_episodes
        self.action_bound_range = 1
        self.plot = False
        self.dflt_dtype = dtype
        self.actor_optim = ko.Adam(self.actor_lr)
        self.critic_optim = ko.Adam(self.critic_lr)
        self.r, self.l, self.qlss, = [], [], []
        self.obs_min = self.env.observation_space.low
        self.obs_max = self.env.observation_space.high
        action_dim = 1
        state_dim = len(env.reset())
        print("State dim = ", state_dim)
        print("action dim = ", action_dim)
        self.buffer = Replay_Buffer(max_buffer_size, batch_size, dtype)
        self.actor = _actor_network(state_dim, action_dim, self.action_bound_range).model()
        self.critic = _critic_network(state_dim, action_dim).model()
        self.actor_target = _actor_network(state_dim, action_dim, self.action_bound_range).model()
        self.actor_target.set_weights(self.actor.get_weights())
        self.critic_target = _critic_network(state_dim, action_dim).model()
        self.critic.compile(loss='mse', optimizer=self.critic_optim)
        self.critic_target.set_weights(self.critic.get_weights())

    
    def take_action(self, state, rand):
        act_n = self.actor.predict(state).ravel()[0]
        if rand:
            return act_n + random.uniform(-1, 1)
        else:
            return act_n

    def train_networks(self, s_batch, a_batch, r_batch, n_s_batch, done_batch, indices=None):
        next_actions = self.actor_target(n_s_batch)

        Q_target_pls_1 = self.critic_target([n_s_batch, next_actions])
        y_i = r_batch
        for i in range(self.batch_size):
            if not done_batch[i]:
                y_i[i] += Q_target_pls_1[i] * self.gamma
        self.critic.train_on_batch([s_batch, a_batch], y_i)


        with tf.GradientTape() as tape:
            a = self.actor(s_batch)
            tape.watch(a)
            q = self.critic([s_batch, a])
        dq_da = tape.gradient(q, a)

        with tf.GradientTape() as tape:
            a = self.actor(s_batch)
            theta = self.actor.trainable_variables
        da_dtheta = tape.gradient(a, theta, output_gradients=-dq_da)
        self.actor_optim.apply_gradients(zip(da_dtheta, self.actor.trainable_variables))

    def update_target(self, target, online, tau):
        init_weights = online.get_weights()
        update_weights = target.get_weights()
        weights = []
        for i in tf.range(len(init_weights)):
            weights.append(tau * init_weights[i] + (1-tau) * update_weights[i])
        target.set_weights(weights)
        return target

    def train(self):
        obs = self.env.reset()
        state_dim = len(obs)
        experience_cnt = 0
        self.ac = []
        rand = True
        for epi in range(self.n_episodes):
            ri, li, qlssi = [], [], []
            state_t = np.array(self.env.reset(), dtype='float32').reshape(1, state_dim)
            state_t = (state_t - self.obs_min)/(self.obs_max-self.obs_min)
            for t in range(self.T):
                action_t = self.take_action(state_t, rand)
                self.ac.append(action_t)
                tmp = self.env.step([action_t])
                s_t_pls_1, r_t, done_t = tmp[0], tmp[1], tmp[2]
                s_t_pls_1 = (s_t_pls_1 - self.obs_min) / (self.obs_max - self.obs_min)
                ri.append(r_t)
                self.buffer.store(
                    state_t.ravel(), action_t, r_t, np.array(s_t_pls_1, 'float32'), done_t
                )

                state_t = np.array(s_t_pls_1, dtype='float32').reshape(1, state_dim)
                if not rand:
                    s_batch, a_batch, r_batch, n_s_batch, done_batch = self.buffer.sample_batch()
                    self.train_networks(s_batch, a_batch, r_batch, n_s_batch, done_batch, None)
                    self.actor_target = self.update_target(self.actor_target, self.actor, self.tau)
                    self.critic_target = self.update_target(self.critic_target, self.critic, self.tau)

                if done_t or t == self.T-1:
                    rr=np.sum(ri)
                    self.r.append(rr)
                    print('Episode %d : Total Reward = %f' % (epi, rr))
                    break
                
                if rand:
                    experience_cnt += 1
                if experience_cnt > self.explore_time :
                    rand = False


if __name__ == "__main__":
    env = gym.make('Pendulum-v0')
    ddpg = DDPG(env = env)# epochs to save models and buffer
    ddpg.train()
