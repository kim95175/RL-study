import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

#HyperParameters
LEARNING_RATE = 0.0002
GAMMA = 0.98

class Policy(nn.Module):
    def __init__(self, state_size, action_size):
        super(Policy, self).__init__()
        self.data = []

        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, action_size)
        self.optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=0)
        return x

    def put_data(self, item):
        self.data.append(item)
    
    def train_net(self):
        R = 0
        self.optimizer.zero_grad()
        for r, prob in self.data[::-1]:
            R = r + GAMMA * R
            loss = -torch.log(prob) * R
            loss.backward()
        self.optimizer.step()
        self.data = []



def main():
    env = gym.make('MountainCar-v0')
    #env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    print(env.observation_space.shape[0])
    print(env.action_space.n)
    pi = Policy(state_size=state_size, action_size=action_size)
    score = 0.0
    print_interval = 500

    for n_epi in range(10000):
        s = env.reset()
        done = False

        while not done:
            if n_epi%print_interval == 0 and n_epi != 0:
                env.render()
            prob = pi(torch.from_numpy(s).float())
            #print("prob : ", prob)
            m = Categorical(prob)
            a = m.sample()
            next_s, r, done, _ = env.step(a.item())
            pi.put_data((r,prob[a]))
            s = next_s
            score += r

        pi.train_net()

        if n_epi%print_interval == 0 and n_epi != 0:
            print("# of episode :{}, avg score : {}".format(n_epi, score/print_interval))
            score = 0.0

    env.close()


if __name__ == '__main__':
    main()