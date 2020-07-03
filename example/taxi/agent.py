import numpy as np
from collections import defaultdict

class Agent:
    def __init__(self, Q, mode="default", nA=6, alpha = 0.01, gamma = 0.99):
        self.Q = Q
        self.mode = mode
        self.nA = nA
        self.alpha = alpha
        self.gamma = gamma
        # Q-table, returns_sum, returns_count 초기화
        if(mode != "default"):
            self.returns_sum = defaultdict(lambda: np.zeros(6))
            self.returns_count = defaultdict(lambda: np.zeros(6))
            for state in range(500):
                p = {} 
                for action in range(self.nA):
                    p[action] = 0
                    self.returns_sum[state][action] = 0
                    self.returns_count[state][action] = 0
                self.Q[state] = p
            self.episode = []
        
        

    def select_action(self, state, eps):
        """
        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """

        #####################################
        # replace this with your code !!!!!!!
        '''
        if( np.random.rand() < eps*8):
            action = np.random.choice(self.nA)
        else:
            #action = max(self.Q[state].items(), key = operator.itemgetter(1))[0]
            action = np.random.choice([key for key in self.Q[state].keys() if self.Q[state][key] == max(self.Q[state].values())])
        '''

        # 추가적인 exploration을 위해 eps 값 추가
        eps = eps*8   
        if(eps >= 1):
            action = np.random.choice(self.nA)
        else:
            A = np.ones(self.nA, dtype=float) * ( eps / self.nA )
            best_a_key = [key for key in self.Q[state].keys() if self.Q[state][key] == max(self.Q[state].values())]
            for i in best_a_key:
                A[i] += (1.0 - eps) / (len(best_a_key))
            action = np.random.choice(np.arange(len(A)), p=A)

        ####################################
        return action


    def step(self, state, action, reward, next_state, done):

        """
        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        # Q-learning 
        if(self.mode == "q_learning"):
            prev_q = self.Q[state][action]
            next_q = max(self.Q[next_state].values())
            self.Q[state][action] = prev_q + self.alpha * (reward + (self.gamma * next_q) - prev_q)

        # First Visit MC Control
        if(self.mode == "mc_control"):
            # 벽에 부딪혀 고립되는 상황을 막기 위해 추가 -reward 부여
            if(next_state == state):
                reward = -4
            self.episode.append((state, action, reward))
            if done:
                G=0
                # episode 내 각 step을 역순으로 올라가며 Retrun값 update
                for i in reversed(range(len(self.episode))):
                    s, a, r = self.episode[i]
                    s_a = (s, a)
                    prev_g = G
                    G = ( prev_g * self.gamma ) + r
                    # First Visit : 현재 step기준 앞 쪽에 동일한 state-action fair가 없을 경우
                    if( (s_a in [(x[0], x[1]) for x in self.episode[:i]]) == False ):
                        self.returns_count[s][a] = self.returns_count[s][a] + 1
                        self.returns_sum[s][a] = self.returns_sum[s][a] + G
                        self.Q[s][a] = self.returns_sum[s][a] / self.returns_count[s][a]
                self.episode = []

