import operator
import numpy as np
Q = {}
state =1
p = {} 
for action in range(6):
    p[action] = 0
    if action == 5:
        p[action] = 10
p[4] = 10
Q[state] = p
action = max(Q[state].items(), key = operator.itemgetter(1))
print( max(Q[state].values()) )
x =  [ key for key in Q[state].keys() if Q[state][key] == max(Q[state].values())]
print( x ) 
print(np.random.choice(x))
#action = np.random.choice([key for key in Q[state].keys() if(Q[state][key] == max(Q[state].items()).any)])
print(Q)
print(action)