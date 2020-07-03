import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.distributions import Categorical

LEARNING_RATE = 0.0002
GAMMA = 0.98
n_rollout = 20

class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size):
        super(ActorCritic, self).__init__()
        self.data = []

        self.fc1 = nn.Linear(state_size, 256)
        self.fc_pi = nn.Linear(256, action_size)
        self.fc_v = nn.Linear(256,1)
        self.optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE)

    def pi(self, x, softmax_dim = 0):
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob

    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v

    def put_data(self, transition):
        self.data.append(transition)
    
    def make_batch(self):
        s_list, a_list, r_list, next_s_list, done_list = [], [], [], [], []
        for transition in self.data:
            s, a, r, next_s, done = transition
            s_list.append(s)
            a_list.append([a])
            r_list.append([r/100.0])
            next_s_list.append(next_s)
            done_mask = 0.0 if done else 1.0
            done_list.append([done_mask])

        s_batch, a_batch, r_batch, next_s_batch, done_batch = torch.tensor(s_list, dtype=torch.float), torch.tensor(a_list), \
                                                               torch.tensor(r_list, dtype=torch.float), torch.tensor(next_s_list, dtype=torch.float), \
                                                               torch.tensor(done_list, dtype=torch.float)
        
        self.data=[]
        return s_batch, a_batch, r_batch, next_s_batch, done_batch
    
    def train_net(self):
        s, a, r, next_s, done = self.make_batch()
        td_target = r + GAMMA * self.v(next_s) * done
        delta = td_target - self.v(s)

        pi = self.pi(s, softmax_dim=1)
        pi_a = pi.gather(1, a)
        loss = -torch.log(pi_a) * delta.detach() + F.smooth_l1_loss(self.v(s), td_target.detach())

        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()

def main():  
    #env = gym.make('CartPole-v1')
    env = gym.make('MountainCar-v0')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    print(state_size, action_size)
    model = ActorCritic(state_size=state_size, action_size=action_size)    
    print_interval = 20
    score = 0.0

    for n_epi in range(10000):
        done = False
        s = env.reset()
        while not done:
            if n_epi%100==0 and n_epi!=0:
                env.render()
            for t in range(n_rollout):
                prob = model.pi(torch.from_numpy(s).float())
                m = Categorical(prob)
                a = m.sample().item()
                s_prime, r, done, info = env.step(a)
                model.put_data((s,a,r,s_prime,done))
                
                s = s_prime
                score += r
                
                if done:
                    break                     
            
            model.train_net()
            
        if n_epi%print_interval==0 and n_epi!=0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, score/print_interval))
            score = 0.0
    env.close()


if __name__ == '__main__':
    main()