import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.layers as kl
import tensorflow.keras.optimizers as ko
import tensorflow.keras as k
import gym

class Memory:
    # initialized by state_n, action_n, BUFFER_SIZE
    def __init__(self, obs_dim, act_dim, size):
        self.states = np.zeros([size, obs_dim], dtype=np.float32)
        self.actions = np.zeros([size, act_dim], dtype=np.float32)
        self.rewards = np.zeros([size, act_dim], dtype=np.float32)
        self.next_states = np.zeros([size, obs_dim], dytpe=np.float32)
        self.dones = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, state, action, reward, next_state, done):
        self.states[self.ptr] = state
        self.next_states[self.ptr] = next_state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def get_sample(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return self.states[idxs], self.actions[idxs], self.rewards[idxs], self.next_states[idxs], self.dones[idxs]


class DDPGAgent:
    def __init__(self, state_dim, action_n, act_limit):
        
        self.action_n = action_n
        self.state_dim = state_dim
        self.state_n = state_dim[0]

        self.ACT_LIMIT = act_limit
        self.GAMMA = 0.99
        self.TAU = 0.005
        self.BUFFER_SIZE = int(1e6)
        self.BATCH_SIZE = 100
        self.ACT_NOISE_SCALE = 0.1

        self.actor = self._generate_actor()
        self.actor_target = self._genrate_actor()
        self.critic = self._generate_critic()
        self.critic_target = self._genrate_critic()

        self.memory = Memory(self.state_n, self.action_n, self.BUFFER_SIZE)

        self.dummy_Q_target_prediction_input = np.zeors((self.BATCH_SIZE, 1))
        self.dummy_dones_input = np.zeros((self.BATCH_SIZE, 1))

    def _generate_actor(self):
        state_input = kl.input(shape=self.state_dim)
        dense = kl.dense(400, activation='relu')(state_input)
        dense2 = kl.dense(300, activation='relu')(dense)
        out = kl.dense(self.action_n, activation='tanh')(dense2)
        model = k.Model(inputs=state_input, outputs=out)
        model.compile(optimizer = 'adam', loss = self._ddpg_actor_loss)
        model.summary()
        return model

    def _ddpg_actor_loss(self, y_true, y_pred):
        return -k.backend.mean(y_true)

    def get_action(self, states, noise=None):
        if noise is None: noise = self.ACT_NOISE_SCALE
        if len(states.shape) == 1: states= states.reshape(1,-1)
        action = self.actor.predict_on_batch(states)
        if noise != 0:
            action += noise * np.random.randn(self.action_n)
            action = np.clip(action, -self.ACT_LIMIT, self.ACT_LIMIT)
        return action
    
    def get_target_action(self, states):
        return self.actor_target.predict_on_batch(states)

    def train_actor(self, states, actions):
        actions_predict = self.get_action(states, noise=0)
        Q_predictions = self.get_Q(states, actions_predict)
        self.actor.train_on_batch(states, Q_predictions)

    def _generate_critic(self):
        state_input = kl.Input(shape=self.state_dim, name='state_input')
        action_input = kl.Input(shape=(self.action_n,), name='action_input')
        Q_target_prediction_input = kl.Input(shape=(1,), name='Q_target_prediction_input')
        dones_input = kl.Input(shape=(1,), name='donse_input')

        concate_state_action = kl.concatenate([state_input, action_input])
        dense = kl.Dense(400,activation='relu')(concate_state_action)
        dense2= kl.Dense(300, activation='relu')(dense)
        out = kl.Dense(1, activation='linear')(dense2)
        model = k.Model(input=[state_input, action_input, Q_target_prediction_input, dones_input], outputs=out)
        model.compile(optimizer='adam', loss= self._ddpg_critic_loss(Q_target_prediction_input, dones_input))
        model.summary()
        return model
    
    def _ddpg_critic_loss(self, Q_target_prediction_input, dones_input):
        def loss(y_true, y_pred):
            target_Q = y_true + (self.GAMMA * Q_target_prediction_input * (1-dones_input))
            mse = k.losses.mse(target_Q, y_pred)
            return mse
        return loss

    def train_critic(self, states, next_states, actions, rewards, dones):
        next_actions = self.get_target_action(next_states)
        Q_target_prediction = self.get_target_Q(next_states, next_actions)
        self.critic.train_on_batch([states, actions, Q_target_prediction, dones], rewards)
    
    def get_Q(self, states, actions):
        return self.critic.predict([states, actions, self.dummy_Q_target_prediction_input,self.dummy_dones_input])

    def get_target_Q(self, states, actions):
        return self.critic_target.predict_on_batch([states, actions, self.dummy_Q_target_prediction_input, self.dummy_dones_input])

    
    def _soft_update_actor_and_critic(self):
        weights_critic_local = np.array(self.critic.get_weights())
        weights_critic_target = np.array(self.critic_target.get_weights())
        self.critic.set_weights(self.TAU * weights_critic_local + (1.0 - self.TAU) * weights_critic_target)
        weights_actor_local = np.array(self.actor.get_weights())
        weights_actor_target = np.array(self.actor_target.get_weights())
        self.actor.set_weights(self.TAU * weights_actor_local + (1.0 - self.TAU) * weights_actor_target)

    def store(self, state, action, reward, next_states, done):
        self.memory.store(state, action, reward, next_states, done)

    def train(self):
        states, actions, rewards, next_states, dones = self.memory.get_sample(batch_size= self.BATCH_SIZE)
        self.train_critic(states, next_states, actions, dones)
        self.train_actor(states, actions)
        self._soft_update_actor_and_critic()


if __name__ == "__main__":
    GAMMA = 0.99
    EPOCHS = 1000
    MAX_EPISODE_LEN = 3000
    START_STEPS = 10000
    RENDER_EVERY = 10

    env = gym.make('LunarLander-v2')
    agent = DDPGAgent(env.observation_space.shape, env.action_space.shape[0], max(env.action_space.high))

    state, reward, done, ep_reward, ep_len, ep_cnt = env.reset(), 0, False, [0,0], 0, 0
    total_steps = MAX_EPISODE_LEN * EPOCHS

    for t in range(total_steps):
        if ep_cnt % RENDER_EVERY == 0:
            env.render()
        
        if t > START_STEPS :
            action = agent.get_action(state)
            action = np.squeeze(action)
        else:
            action = env.action_space.sample()
        
        next_state, reward, done, _ = env.step(action)
        ep_reward[-1] += reward
        ep_len +=1

        done = False if ep_len==MAX_EPISODE_LEN else done

        agent.store(state, action, reward, next_state, done)

        state = next_state

        if done or (ep_len == MAX_EPISODE_LEN):
            ep_cnt += 1
            if True:
                print(f"Episode: {len(ep_reward)-1}, Reward: {np.mean(ep_reward[-12:-2])}")

            ep_reward.append(0.0)

            for _ in range(ep_len):
                agent.train()

            state, reward, done, ep_reward, ep_len = env.reset(), 0, False, 0, 0


    SMA_rewards = np.convolve(ep_reward, np.ones((5,))/5, mode='valid')
    plt.style.use('seaboarn')
    plt.plot(SMA_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.show()