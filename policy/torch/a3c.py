import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import torch.multiprocessing as mp
import time

n_train_processes = 3
LEARNING_RATE = 0.0002
update_interval = 5
GAMMA = 0.98
max_train_ep = 5000
max_test_ep = 6000


class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.fc_pi = nn.Linear(256, action_size)
        self.fc_v = nn.Linear(256, 1)

    def pi(self, x, softmax_dim=0):
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob
    
    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v

def train(global_model, rank):
    env = gym.make('MountainCar-v0')
    #env = gym.make('CartPole-v1')
    #state_size = env.observation_space.shape[0]
    #action_size = env.action_space.n
    #print(env.observation_space.shape[0])
    #print(env.action_space.n)
    state_size = 2
    action_size = 3

    local_model = ActorCritic(state_size=state_size, action_size=action_size)
    local_model.load_state_dict(global_model.state_dict())

    optimizer = optim.Adam(global_model.parameters(), lr=LEARNING_RATE)

    for n_epi in range(max_train_ep):
        done = False
        s = env.reset()
        while not done:
            s_list, a_list, r_list = [], [], []
            for t in range(update_interval):
                prob = local_model.pi(torch.from_numpy(s).float())
                m = Categorical(prob)
                #print("m : ", m)
                a = m.sample().item()
                next_s, r, done, _ = env.step(a)

                s_list.append(s)
                a_list.append([a])
                r_list.append(r/100.0)

                s = next_s
                if done:
                    break

            s_final = torch.tensor(next_s, dtype=torch.float)
            R = 0.0 if done else local_model.v(s_final).item()
            td_target_list = []
            for reward in r_list[::-1]:
                R = GAMMA * R + reward
                td_target_list.append([R])
            td_target_list.reverse()

            s_batch, a_batch, td_target = torch.tensor(s_list, dtype = torch.float), \
                                            torch.tensor(a_list), \
                                            torch.tensor(td_target_list)
            
            advantage = td_target - local_model.v(s_batch)

            pi = local_model.pi(s_batch, softmax_dim=1)
            pi_a = pi.gather(1, a_batch)
            loss = -torch.log(pi_a) * advantage.detach() + \
                F.smooth_l1_loss(local_model.v(s_batch), td_target.detach())
            
            optimizer.zero_grad()
            loss.mean().backward()
            for global_param, local_param in zip(global_model.parameters(), local_model.parameters()):
                global_param._grad = local_param.grad
            optimizer.step()
            local_model.load_state_dict(global_model.state_dict())
        
    
    env.close()
    print("Training process {} reached maximum episode.".format(rank))

def test(global_model):
    env = gym.make('MountainCar-v0')
    score = 0.0
    print_interval = 20

    for n_epi in range(max_test_ep):
        done = False
        s = env.reset()
        while not done:
            prob = global_model.pi(torch.from_numpy(s).float())
            a = Categorical(prob).sample().item()
            next_s, r, done, _ = env.step(a)
            s = next_s
            score += r
        
        if n_epi % print_interval == 0 and n_epi != 0:
            print("# of episode :{}, avg score : {:.1f}".format(
                n_epi, score/print_interval))
            score = 0.0
            time.sleep(1)
    env.close()

if __name__ == '__main__':
    global_model = ActorCritic(state_size=2, action_size=3)
    global_model.share_memory()

    processes = []
    for rank in range(n_train_processes +1):
        if rank == 0:
            p = mp.Process(target=test, args=(global_model,))
        else:
            p = mp.Process(target=train, args=(global_model, rank,))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    
    
                                            