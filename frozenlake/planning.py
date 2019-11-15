import main_lib
import numpy as np
from frozenlake import FrozenLakeEnv

while True:
    mode = int(input("1.Not Slippery, 2.Slippery: "))
    if mode == 1:
        env = FrozenLakeEnv(is_slippery=False)
        break
    elif mode == 2:
        env = FrozenLakeEnv(is_slippery=True)
        break
    else:
        print("잘못 입력했습니다.")

# 환경 state 개수 및 action 개수
#print(env.nS, env.nA)

while True:
    mode = int(input("1.Policy Iteration, 2.Value Iteration: "))
    if mode == 1:
        policy, V = main_lib.policy_iteration(env)
        break
    elif mode == 2:
        policy, V = main_lib.value_iteration(env)
        break
    else:
        print("잘못 입력했습니다.")
print()


# state 0 에서 action 1 을 선택했을 때 [상태 이동 확률, 도착 state, reward]
'''
for i in range(env.nS):
    print(env.MDP[i])
'''

print("Optimal State-Value Function:")
for i in range(len(V)):
    if i>0 and i%env.ncol==0:
        print()
    print('{0:0.3f}'.format(V[i]), end="\t")
print("\n")

print("Optimal Policy [LEFT, DOWN, RIGHT, UP]:")
action = {0:"LEFT", 1:"DOWN", 2:"RIGHT", 3:"UP"}
col = env.ncol
for i in range(len(policy)):
    if i>0 and i%(col)==0:
        print()
    tmprow, tmpcol = i//col, i%col
    #print(env.map[tmprow, tmpcol], end = '  ')
    print(np.argwhere(policy[i] != 0), end='    ')
print()