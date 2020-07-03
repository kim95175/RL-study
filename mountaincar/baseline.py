import gym
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines import DQN
from collections import deque

dqn, ddqn, per, duel, acer = False, False, False, False, False  

if len(sys.argv) == 1:
    # Default DQN
    print("There is no argument, please input")

for i in range(1,len(sys.argv)):
    if sys.argv[i] == "dqn":
        dqn = True
    elif sys.argv[i] == "ddqn":
        ddqn = True
    elif sys.argv[i] == "per":
        per = True
    elif sys.argv[i] == "duel":
        duel = True
    elif sys.argv[i] == "acer":
        acer = True

def test(model):
    episode_record = []     # plot을 그리기 위해 데이터를 저장하는 list
    last_100_game_reward = deque(maxlen=100)
    max_episode = 10
    
    for i in range(max_episode):
        state = env.reset()
        done = False
        step_count = 0
        episode_reward = 0

        while not done:
            env.render()
            action, _states = model.predict(state)
            next_state, rewards, done, info = env.step(action)
            state = next_state
            step_count += 1
            episode_reward += rewards


        last_100_game_reward.append(episode_reward)
        avg_reward = np.mean(last_100_game_reward)
        episode_record.append(avg_reward)
        print("[Episode {:>5}]  episode steps: {:>5} avg: {}".format(i, step_count, avg_reward))

    return episode_record


env = gym.make('MountainCar-v0')

#training model
if dqn:
    env.reset()
    model = DQN(MlpPolicy, env, verbose=1, double_q=False)
    model.learn(total_timesteps=38000)
    dqns = test(model)
    del model
if ddqn:
    env.reset()
    model = DQN(MlpPolicy, env, verbose=1, double_q=True)
    model.learn(total_timesteps=38000)
    ddqns = test(model)
    del model    
if per:
    env.reset()
    model = DQN(MlpPolicy, env, verbose=1, prioritized_replay=True)
    model.learn(total_timesteps=50000)
    pers = test(model)
    del model




print("Reinforcement Learning Finish")
print("Draw graph ... ")

x = np.arange((10))

if dqn:
    plt.plot(x, dqns, label='DQN')
if ddqn:
    plt.plot(x, ddqns, label='Double')
if per:
    plt.plot(x, pers, label='PER')


plt.legend()
fig =plt.gcf()
plt.savefig("result.png")
plt.show()



