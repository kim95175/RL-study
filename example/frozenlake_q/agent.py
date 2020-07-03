import numpy as np
from collections import defaultdict, deque

class Agent:
    def __init__(self, Q, env, mode, nA=4, alpha = 0.01, gamma = 0.99):
        self.Q = Q
        self.env = env
        self.mode = mode
        self.nA = nA
        self.alpha = alpha
        self.gamma = gamma
        self.eps = 0.5
        print(self.env.observation_space)
        
    def select_action(self, state, eps = 0.000001):
        if(self.mode == "learning_mode"):  
            A = np.ones(self.nA, dtype=float) * ( eps / self.nA )
            best_a_key = [key for key in range(self.nA) if self.Q[state][key] == max(self.Q[state])]
            for i in best_a_key:
                A[i] += (1.0 - eps) / (len(best_a_key))
            action = np.random.choice(np.arange(len(A)), p=A)

            return action
        else:
            action = np.argmax(self.Q[state])
            return action

    def learn(self):
        num_epi = 100000
        sample_rewards = deque(maxlen=100)
        
        for n_epi in range(num_epi):
            eps = 1.0 / ( (n_epi//1000)+1)
            state = self.env.reset()
            #print(self.Q[state])
            sample_reward = 0

            while True: 
                action = self.select_action(state, eps)
                next_state, reward, done, _ = self.env.step(action)

                prev_q = self.Q[state][action]
                next_q = max(self.Q[next_state])
                self.Q[state][action] = prev_q + self.alpha * (reward + (self.gamma * next_q) - prev_q)
                sample_reward += reward
                if done:
                    sample_rewards.append(sample_reward)
                    break
                state = next_state
            if n_epi % 100 == 0 and n_epi != 0:
                avg_reward = sum(sample_rewards) / len(sample_rewards)
                print("\rEpisode {}/{} || average reward {}, eps {:.2f}".format(n_epi, num_epi, avg_reward, eps*100), end="")
