import gym
import sys
import matplotlib.pyplot as plt
import numpy as np
import random
from dqn import DQN

NUM_LEARN = 1500
UP_ACTION = 2
DOWN_ACTION = 3

env = gym.make("Pong-ram-v0")

env.reset()
dqn = DQN(env, double_q=False, per=False, multistep=False)
defaults = dqn.learn(NUM_LEARN)
del dqn

print("Reinforcement Learning Finish")
print("Draw graph ... ")

x = np.arange((NUM_LEARN))
plt.plot(x, defaults, label = 'DQN')

plt.legend()
fig =plt.gcf()
plt.savefig("result.png")
plt.show()


'''
observation = env.reset()

for i in range(300):
    env.render()

    action = random.randint(UP_ACTION, DOWN_ACTION)

    observation, reward, done, _ = env.step(action)

    if done:
        env.reset()
'''