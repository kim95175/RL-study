import tensorflow as tf
print(tf.__version__)

import gym
import time
import numpy as np
import tensorflow.keras.layers as kl
import tensorflow.keras.optimizers as ko

np.random.seed(1)
tf.random.set_seed(1)

class Model(tf.keras.Model):
    def __init__(self, num_actions):
        super(Model, self).__init__(name='basic_dqn')
        self.hidden1 = kl.Dense(32, activation='relu', kernel_initializer='he_uniform')
        self.hidden2 = kl.Dense(32, activation='relu', kernel_initializer='he_uniform')
        self.logits = kl.Dense(num_actions, name='q_values')

    def call(self, inputs):
        x = tf.convert_to_tensor(inputs)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.logits(x)
        return x

    def action_value(self, obs):
        q_values = self.predict(obs)
        best_action = np.argmax(q_values, axis=-1)
        return best_action[0], q_values[0]

def test_model():
    env = gym.make('CartPole-v0')
    print('num_actions: ', env.action_space.n)
    model = Model(env.action_space.n)

    obs = env.reset()
    print('obs_shape: ', obs.shape)

    best_action, q_values = model.action_value(obs[None])
    print('res of test model: ', best_action, q_values)


class DQNAgent:
    def __init__(self, model, target_model, env, buffer_size=100, learning_rate=0.015, epsilon=0.1,
                epsilon_decay=0.995, min_epsilon=0.01, gamma=0.97, batch_size=16, target_update_iter=400,
                train_nums=8000, start_learning=32):
        self.model = model
        self.target_model = target_model
        opt = ko.Adam(learning_rate=learning_rate, clipvalue=10.0)
        self.model.compile(optimizer=opt, loss='mse')

        #param
        self.env = env
        self.lr = learning_rate
        self.epsilon = epsilon
        self.epsioln_dacay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_iter = target_update_iter
        self.train_nums = train_nums
        self.num_in_buffer = 0
        self.buffer_size = buffer_size
        self.start_learning = start_learning

        # replay buffer params [(s, a, r, ns, done), ...]
        self.obs = np.empty((self.buffer_size,) + self.env.reset().shape)
        self.actions = np.empty((self.buffer_size), dtype=np.int8)
        self.rewards = np.empty((self.buffer_size), dtype=np.float32)
        self.dones = np.empty((self.buffer_size), dtype=np.bool)
        self.next_states = np.empty((self.buffer_size,) + self.env.reset().shape)
        self.next_idx = 0

    def train(self):
        obs = self.env.reset()
        for t in range(1, self.train_nums):
            best_action, q_valuse = self.model.action_value(obs[None])
            action = self.get_action(best_action)
            next_obs, reward, done, _ = self.env.step(action)
            self.store_transition(obs, action, reward, next_obs, done)
            self.num_in_buffer = min(self.num_in_buffer + 1, self.buffer_size)

            if t > self.start_learning:
                losses = self.train_step()
                if t % 1000 == 0:
                    print('losses each 1000 steps: ', losses)

            if t % self.target_update_iter == 0:
                self.update_target_model()
            if done:
                obs = self.env.reset()
            else:
                obs = next_obs  

    def train_step(self):
        idxes = self.sample(self.batch_size)
        s_batch = self.obs[idxes]
        a_batch = self.actions[idxes]
        r_batch = self.rewards[idxes]
        ns_batch = self.next_states[idxes]
        done_batch = self.dones[idxes]

        target_q = r_batch + self.gamma * np.amax(self.get_target_value(ns_batch), axis=1) * (1 - done_batch)
        target_f = self.model.predict(s_batch)
        for i, val in enumerate(a_batch):
            target_f[i][val] = target_q[i]

        losses = self.model.train_on_batch(s_batch, target_f)

        return losses

    def evaluation(self, env, render=True):
        obs, done, ep_reward = env.reset(), False, 0
        while not done:
            action, q_valuse = self.model.action_value(obs[None])
            obs, reward, done, _ = env.step(action)
            ep_reward += reward
            if render:
                env.render()
            time.sleep(0.05)
        env.close()
        return ep_reward

    def store_transition(self, obs, action, reward, next_state, done):
        n_idx = self.next_idx % self.buffer_size
        self.obs[n_idx] = obs
        self.actions[n_idx] = action
        self.rewards[n_idx] = reward
        self.next_states[n_idx] = next_state
        self.dones[n_idx] = done
        self.next_idx = (self.next_idx +1) % self.buffer_size

    def sample(self, n):
        assert n < self.num_in_buffer
        res = []
        while True:
            num = np.random.randint(0, self.num_in_buffer)
            if num not in res:
                res.append(num)
            if len(res) == n: 
                break
        return res

    def get_action(self, best_action):
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        return best_action

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
    
    def get_target_value(self, obs):
        return self.target_model.predict(obs)
    
    def e_decay(self):
        self.epsilon *= self.epsioln_dacay

if __name__ == '__main__':
    test_model()

    env = gym.make("CartPole-v0")
    num_actions = env.action_space.n
    model = Model(num_actions)
    target_model = Model(num_actions)
    agent = DQNAgent(model, target_model,  env)
    # test before
    rewards_sum = agent.evaluation(env)
    print("Before Training: %d out of 200" % rewards_sum) # 9 out of 200

    agent.train()
    # test after
    rewards_sum = agent.evaluation(env)
    print("After Training: %d out of 200" % rewards_sum) # 200 out of 200

