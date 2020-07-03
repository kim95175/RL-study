import gym
from gym.envs.registration import register
from collections import deque
from collections import defaultdict
import numpy as np
from agent import Agent

is_slippery = input("is_slippery no or yes : ")
is_slippery = "yes"
if is_slippery == "yes":
    is_slippery = True
else:
    is_slippery = False

#map_size = input("map_size 4x4 or 8x8 : ")

map_size = "4x4"

register(
        id='FrozenLake-v3',
        entry_point='gym.envs.toy_text:FrozenLakeEnv',
        kwargs={'map_name' : map_size, 'is_slippery': is_slippery}
    )

env = gym.make('FrozenLake-v3')
action_size = env.action_space.n

def check_environement():
    env.reset()
    env.render()
    print()
    while True:
        action = input("Enter action: ")
        if action not in ['0','1','2','3']:
            continue
        action = int(action)
        state, reward, done, info = env.step(action)
        env.render()
        print("State :", state, "Action: ", action, "Reward: ", reward, "info: ", info)
        print()
        if done:
            print("Finished with reward", reward)
            break


def testing_after_learning(Q):
    agent = Agent(Q, env, "testing_mode")
    total_test_episode = 1000
    rewards = []
    for episode in range(total_test_episode):
        state = env.reset()
        episode_reward = 0
        while True:
            action = agent.select_action(state)
            new_state, reward, done, _ = env.step(action)
            episode_reward += reward
            if done:
                rewards.append(episode_reward)
                break
            state = new_state
    print("avg: " + str(sum(rewards) / total_test_episode))

Q = defaultdict(lambda: np.zeros(action_size))
while True:
    print()
    print("1. Checking Frozen_Lake")
    print("2. Q-learning")
    print("3. Testing after learning")
    print("4. Exit")
    menu = int(input("select: "))
    print(action_size)
    if menu == 1:
        check_environement()
    elif menu == 2:
        Q = defaultdict(lambda: np.zeros(action_size))
        agent = Agent(Q, env, "learning_mode")
        agent.learn()
    elif menu == 3:
        testing_after_learning(Q)
    elif menu == 4:
        break
    else:
        print("wrong input!")
