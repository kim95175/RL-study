import numpy as np
import tensorflow.compat.v1 as tf
import gym
from gym.utils import colorize


def dense_nn(inputs, layers_sizes, name="mlp", reuse=False, output_fn=None,
            dropout_keep_prob=None, batch_norm=False, training=True):
    
    print(colorize("Building mlb {} | sizes: {}".format(
        name, [inputs.shape[0]] + layers_sizes), "green"
    ))

    with tf.variable_scope(name, reuse=reuse):
        out = inputs
        for i, size in enumerate(layers_sizes):
            print("Layer:", name + '_l' + str(i), size)
            if i > 0 and dropout_keep_prob is not None and training:
                out = tf.nn.dropout(out, dropout_keep_prob)

            out = tf.layers.dense(
                out,
                size,
                activation=tf.nn.relu if i < len(layers_sizes) - 1 else None,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                name=name + '_l' + str(i),
                reuse=reuse
            )

            if batch_norm:
                out = tf.layers.batch_normalization(out, training = training)

        if output_fn:
            out = output_fn(out)

    return out 

class Policy:
    def __init__(self, env, name, training=True, gamma=0.99, deterministic=False):
        self.env = env
        self.gamma = gamma
        self.training = training
        self.name = name

        if deterministic:
            np.random.seed(1)
            tf.set_random_seed(1)

    @property
    def act_size(self):
        # number of options of an action; this only makes sense for discrete actions.
        if isinstance(self.env.action_space, Discrete):
            return self.env.action_space.n
        else:
            return None

    @property
    def act_dim(self):
        # dimension of an action; this only makes sense for continuous actions.
        if isinstance(self.env.action_space, Box):
            return list(self.env.action_space.shape)
        else:
            return []

    @property
    def state_dim(self):
        # dimension of a state.
        return list(self.env.observation_space.shape)

    def obs_to_inputs(self, ob):
        return ob.flatten()

    def act(self, state, **kwargs):
        pass

    def build(self):
        pass

    def train(self, *args, **kwargs):
        pass

    def evaluate(self, n_episodes):
        reward_history = []
        reward = 0.

        for i in range(n_episodes):
            ob = self.env.reset()
            done = False
            while not done:
                a = self.act(ob)
                new_ob, r, done, _ = self.env.step(a)
                self.env.render()
                reward += r
                ob = new_ob

            reward_history.append(reward)
            reward = 0.

        print("Avg. reward over {} episodes: {:.4f}".format(n_episodes, np.mean(reward_history)))

class DQN(Policy, BaseModelMixin):
    def __init__(self, env, name,
                training=True,
                gamma=0.99,
                batch_size=32,
                model_type='dense',
                step_size=1,
                layer_sizes=[32,32],
                double_q=False,
                dueling=False):
        
