import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as kl
import tensorflow.keras.losses as kls
import tensorflow.keras.optimizers as ko
import gym
import logging
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-b', '--batch_size', type=int, default=8)
parser.add_argument('-n', '--num_updates', type=int, default=200)
parser.add_argument('-lr', '--learning_rate', type=float, default=7e-3)
parser.add_argument('-r', '--render_test', action='store_true', default=False)
parser.add_argument('-p', '--plot_results', action='store_true', default=True)

class ProbabilityDistribution(tf.keras.Model):
    def call(self, logits, **kwargs):
        # Sample a random categorical action from the given logits.
        return tf.squeeze(tf.random.categorical(logits, 1), axis=-1)


class Model(tf.keras.Model):
    def __init__(self, num_actions):
        super().__init__('mlp_policy')
        self.hidden1 = kl.Dense(128, activation='relu')
        self.hidden2 = kl.Dense(128, activation='relu')
        self.value = kl.Dense(1, name='value')
        self.logits = kl.Dense(num_actions, name='policy_logits')
        self.dist = ProbabilityDistribution()

    def call(self, inputs, **kwargs):
        x = tf.convert_to_tensor(inputs)
        hidden_logs = self.hidden1(x)
        hidden_vals = self.hidden2(x)
    
        return self.logits(hidden_logs), self.value(hidden_vals)
    
    def action_value(self, obs):
        logits, value = self.predict_on_batch(obs)
        action = self.dist.predict_on_batch(logits)
        return np.squeeze(action, axis=-1), np.squeeze(value, axis=-1)


class A2CAgent:
    def __init__(self, model, lr=0.007, gamma=0.99, vf_coef=0.5, ent_coef=1e-4):
        # Coefficients are used for the loss terms.
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        
        # Discount Factor
        self.gamma = gamma
    
        self.model = model
        self.model.compile(
            optimizer=ko.RMSprop(lr=lr),
            # Define seperate losses for policy logist and value estimate
            loss=[self._logits_loss, self._value_loss]
        )
        
    def train(self, env, batch_size=64, updates=250):
        # Storage helpers for a single batch of data.
        actions = np.empty((batch_size,), dtype=np.int32)
        rewards, dones, values = np.empty((3, batch_size))
        observations = np.empty((batch_size,) + env.observation_space.shape)

        print("actions : ", actions.shape)
        print("rewards : ", rewards.shape)
        print("observations : ", observations.shape)

        # Training loop : collect samples, send to optimizer, repeat updates times.
        ep_rewards = [0.0]
        next_obs = env.reset()
        for update in range(updates):
            for step in range(batch_size):
                observations[step] = next_obs.copy()
                actions[step], values[step] = self.model.action_value(next_obs[None, :])
                next_obs, rewards[step], dones[step], _ = env.step(actions[step])
                ep_rewards[-1] += rewards[step]
                if dones[step]:
                    ep_rewards.append(0.0)
                    next_obs = env.reset()
                    logging.info("Episode: %03d, Reward: %03d" % (
                        len(ep_rewards) - 1, ep_rewards[-2]))
                    #print(ep_rewards)
            
            _, next_value = self.model.action_value(next_obs[None, :])

            returns, advs = self._returns_advantages(rewards, dones, values, next_value)
            # A trick to input actions and advantages through same API
            acts_and_advs = np.concatenate([actions[:, None], advs[:, None]], axis=-1)

            # Performs a full training step on the collected batch
            # Note : No need to mess around with gradients, Keras API handles it.
            losses = self.model.train_on_batch(observations, [acts_and_advs, returns])

            logging.debug("[%d/%d] losses: %s" %(update+1, updates, losses))
        
        return ep_rewards
    
    def _returns_advantages(self, rewards, dones, values, next_value):
        print("rewards ---", rewards)
        print("next_value ----", next_value)
        # 'next_value' is the bootstrap value estimate of the future state (critic).
        returns = np.append(np.zeros_like(rewards), next_value, axis=-1)
        # Returns are calculated as discounted sum of future rewards.
        for t in reversed(range(rewards.shape[0])):
            returns[t] = rewards[t] + self.gamma * returns[t+1] * (1-dones[t])
 
        returns = returns[:-1]


        # Advantages are equal to returns - baseline (value estimates in our case).
        advantages = returns - values

        return returns, advantages

    def test(self, env, render=False):
        obs, done, ep_reward = env.reset(), False, 0
        while not done:
            action, _ = self.model.action_value(obs[None, :])
            obs, reward, done, _ = env.step(action)
            ep_reward += reward
            if render:
                env.render()
        return ep_reward
    
    def _value_loss(self, returns, value):
        # Value loss is typically MSE between value estimates and returns
        return self.vf_coef * kls.mean_squared_error(returns, value)
    
    def _logits_loss(self, actions_and_advantages, logits):
        # A trick to inpiut actions and advantages through the sam API
        actions, advantages = tf.split(actions_and_advantages, 2, axis=-1)

        # Sparse categorical CE loss obj that supports sample_wight args on 'call()'.
        # 'from_logits' argument ensures transformation into normalized probabilities
        weighted_sparse_ce = kls.SparseCategoricalCrossentropy(from_logits=True)
        
        # Policy loss is defined by policy gradients, weighted by advanatages.
        # Note : we only calculate the loos on the actions we've actually taken/
        actions = tf.cast(actions, tf.int32)
        policy_loss = weighted_sparse_ce(actions, logits, sample_weight=advantages)

        # Entropy loss can be calucated as cross-entropy over itself.
        probs = tf.nn.softmax(logits)
        entropy_loss = kls.categorical_crossentropy(probs, probs)

        # We want to minimize policy and maximize entropy losses.
        # Here signs are flipped because the optimizer minimizes.
        return policy_loss - self.ent_coef * entropy_loss


if __name__ == '__main__':

    args = parser.parse_args()
    logging.getLogger().setLevel(logging.INFO)
    env = gym.make('CartPole-v0')


    model = Model(num_actions=env.action_space.n)
    agent = A2CAgent(model, args.learning_rate)
    obs = env.reset()
    action, value = model.action_value(obs[None, :])
    print(action, value)
    rewards_sum = agent.test(env)
    print("%d out of 200" % rewards_sum)

    rewards_history = agent.train(env, args.batch_size, args.num_updates)
    print("Finished training, testing....")
    print("%d out of 200" % agent.test(env, args.render_test))

    if args.plot_results:
        plt.style.use('seaborn')
        plt.plot(np.arange(0, len(rewards_history), 10), rewards_history[::10])
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.show()