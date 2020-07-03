import numpy as np


def policy_evaluation(env, policy, gamma=0.99, theta=1e-8):
    V = np.zeros(env.nS)

    while True:
        delta = 0
        for s in range(env.nS):
            Vs = 0
            for a in range(env.nA):
                pi = policy[s][a]
                if(pi != 0):
                    for prob, newstate, rew in env.MDP[s][a]:
                        Vs += pi * prob * (rew + V[newstate] * gamma)

            delta = max(delta, np.abs(V[s] - Vs))
            V[s] = Vs
    
        if delta < theta:
            break

    return V

def policy_improvement(env, V, gamma=0.99):
    policy = np.zeros([env.nS, env.nA]) / env.nA
    for s in range(env.nS):
        q = np.zeros(env.nA)
        for a in range(env.nA):
            for prob, newstate, rew in env.MDP[s][a]:
                q[a] += prob * (rew + gamma * V[newstate])

        action = np.zeros(env.nA)
        action[np.argmax(q)] += 1
        policy[s] = action  
 
    return policy


def policy_iteration(env, gamma=0.99, theta=1e-8):
    V = np.zeros(env.nS)
    policy = np.ones([env.nS, env.nA]) / env.nA

    print(env.map)

    while True:
        policy_stable = True
        V = policy_evaluation(env, policy)
        new_policy = policy_improvement(env, V)
        policy_stable =(policy_evaluation(env, policy) == policy_evaluation(env, new_policy)).all()

        if(policy_stable):
            break;

        policy = new_policy
       
    return policy, V

def value_iteration(env, gamma=0.99, theta=1e-8):
    V = np.zeros(env.nS)
    policy = np.ones([env.nS, env.nA]) / env.nA
    
    while True:
        delta = 0

        for s in range(env.nS):
            v = V[s]
            q = np.zeros(env.nA)
            for a in range(env.nA):
                for prob, newstate, rew in env.MDP[s][a]:
                    q[a] += prob * (rew + gamma * V[newstate])
            V[s] = max(q)
            delta = max(delta, abs(V[s]-v))
        
        if delta < theta:
            break
    
    policy = policy_improvement(env, V)
        

    return policy, V
