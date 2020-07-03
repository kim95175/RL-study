import gym
import collections
from collections import deque, namedtuple
import random
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

LEARNING_RATE = 0.0005
GAMMA = 0.98
BUFFER_LIMIT = 50000
BATCH_SIZE = 64
INITIAL_EPSILON = 0.1
MIN_EPSILON = 0.01
EPSILON_DECAY = 0.01
TAU = 1e-3
UPDATE_EVERY = 4

class Qnet(nn.Module):
    def __init__(self, state_size, action_size):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def sample_action(self, obs, epsilon):
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0,1)
        else:
            return out.argmax().item()

class PrioritizedReplayBuffer(object):
    """
    https://github.com/higgsfield/RL-Adventure/blob/master/4.prioritized%20dqn.ipynb
    """
    def __init__(self, buffer_size, batch_size, prob_alpha=0.6, prob_beta=0.5):
        self.prob_alpha = prob_alpha
        self.prob_beta = prob_beta
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((buffer_size, ), dtype=np.float32) # long array
        self.experience = namedtuple("Experience", field_names=["state", 
                                                                "action", 
                                                                "reward", 
                                                                "next_state",
                                                                "done"])
        
    def add(self, state, action, reward, next_state, done):
        
        # if self.buffer is empty return 1.0, else max
        max_priority = self.priorities.max() if self.buffer else 1.0
        exp = self.experience(state, action, reward, next_state, done)
        
        # if buffer has rooms left
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(exp)
        else:
            self.buffer[self.pos] = exp
        
        # assign max priority
        self.priorities[self.pos] = max_priority
        
        # update index
        self.pos = (self.pos + 1) % self.buffer_size 
        
    def sample(self, completion):
        
        beta = self.prob_beta + (1-self.prob_beta) * completion
        
        # if buffer is maxed out..
        if len(self.buffer) == self.buffer_size:
            # all priorities are the same as self.priorities
            priorities = self.priorities
        else:
            # all priorities are up to self.pos cuz it's not full yet
            priorities = self.priorities[:self.pos]
            
        # $ P(i) = (p_i^\alpha) / \Sigma_k p_k^\alpha $
        probabilities_a = priorities ** self.prob_alpha
        sum_probabilties_a = probabilities_a.sum()
        P_i = probabilities_a / sum_probabilties_a
        
        sampled_indices = np.random.choice(len(self.buffer), self.batch_size, p=P_i)
        experiences = [self.buffer[idx] for idx in sampled_indices]
        
        # $ w_i = ( 1/N * 1/P(i) ) ** \beta $
        # $ w_i = ( N * P(i) ** (-1 * \beta) ) $
        N = len(self.buffer)
        weights = ( N * P_i[sampled_indices] ) ** (-1 * beta)
        
        #  For stability reasons, we always normalize weights by 1/ maxi wi so
        #  that they only scale the update downwards.
        weights = weights / weights.max()
        
        states = torch.from_numpy(np.vstack([exp.state for exp in experiences if exp is not None])).float()
        actions = torch.from_numpy(np.vstack([exp.action for exp in experiences if exp is not None])).long()
        rewards = torch.from_numpy(np.vstack([exp.reward for exp in experiences if exp is not None])).float()
        next_states = torch.from_numpy(np.vstack([exp.next_state for exp in experiences if exp is not None])).float()
        dones = torch.from_numpy(np.vstack([exp.done for exp in experiences if exp is not None]).astype(np.uint8)).float()
        weights = torch.from_numpy(np.vstack(weights)).float()
        
        return states, actions, rewards, next_states, dones, sampled_indices, weights
        
    def update_priorities(self, batch_indicies, batch_priorities):
        for idx, priority in zip(batch_indicies, batch_priorities):
            self.priorities[idx] = priority
        
    def __len__(self):
        return len(self.buffer)

class DDQNPERAgent():
    
    def __init__(self, state_size, action_size, seed):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        
        # initialize Q-Network
        self.qnetwork_local = Qnet(state_size=self.state_size, action_size=self.action_size)
        self.qnetwork_target = Qnet(state_size=self.state_size, action_size=self.action_size)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LEARNING_RATE)
        
        # replay memory
        self.memory = PrioritizedReplayBuffer(BUFFER_LIMIT, BATCH_SIZE)
        
        # initialize time step
        self.t_step = 0
    
    
    def step(self, state, action, reward, next_state, done, completion):
        
        # save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
    
        # Learn every UPDATE_EVERY time steps
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample(completion)
                self.learn(experiences, GAMMA)
    
    def act(self, state, eps=0.):
        # single state to state tensor (batch size = 1)
        state = torch.from_numpy(state).float().unsqueeze(0)
        
        # set eval mode for local QN 
        self.qnetwork_local.eval()
        
        # predict state value with local QN
        with torch.no_grad(): # no need to save the gradient value
            action_values = self.qnetwork_local(state)
        
        # set the mode of local QN back to train
        self.qnetwork_local.train()
        
        # e-greedy action selection
        # return greedy action if prob > eps
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        
        # return random action if prob <= eps
        else:
            return random.choice(np.arange(self.action_size))
        
    
    def learn(self, experiences, gamma, p_eps = 1e-5):
        
        # unpack epxeriences
        states, actions, rewards, next_states, dones, sampled_indicies, weights = experiences
        best_actions = self.qnetwork_local(next_states).max(1)[1].unsqueeze(1)
        Q_targets_next = self.qnetwork_target(next_states).detach().gather(1, best_actions)
        
        # compute Q targets from current states
        Q_targets = rewards + gamma * Q_targets_next * (1-dones)
        
        # get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        
        TD_Error = Q_targets - Q_expected
        new_priorities = TD_Error.abs().detach().numpy() + p_eps
        
        loss = (TD_Error.pow(2) * weights).mean()
        
        # minimise the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.memory.update_priorities(sampled_indicies, new_priorities)
        self.optimizer.step()
        
        # update target network
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)
    
    def soft_update(self, local_model, target_model, tau):
        # Q_target = TAU * local_model + (1-TAU) * target_model
        
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0 - tau)*target_param.data)


def ddqn_with_per(agent, env, n_episodes=2000, eps_start=1.0, eps_end=0.01, eps_decay=0.995, fname="dqn"):
    
    '''
    output_path = "outputs/{}".format(fname)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    '''
    scores = [] # list containing scores from each episode
    scores_window = deque(maxlen=100) # last 100 scores
    eps = eps_start # initialize epsilon
    save_score_threshold = 200
    done = False
    
    # for every episode..
    for i_episode in range(1, n_episodes + 1):
        completion = i_episode / n_episodes
        state = env.reset()
        score = 0
        while not done:
            action = agent.act(state, eps)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done, completion)
            state = next_state
            score += reward
            if done:
                break
        
        # append the episode score to the deque
        scores_window.append(score)
        # append the episode score to the list
        scores.append(score)
        
        # decrease episilon
        eps = max(eps_end, eps_decay * eps)
        
        # display metrics
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        
        # save model if the latest average score is higher than 200.0
        if np.mean(scores_window) >= save_score_threshold:
            print('\nEnvironment solved in {:d} episodes! \tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), '{}/cp{:04d}.pth'.format(output_path, i_episode-100))
            save_score_threshold += 5
    
    #with open(os.path.join(output_path, "score.pkl"), "wb") as f:
    #    pickle.dump(scores, f)

def main():
    env = gym.make('MountainCar-v0')
    #env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    print(env.observation_space.shape[0])
    print(env.action_space.n)
    env.seed(0)
    agent = DDQNPERAgent(state_size=state_size, action_size=action_size, seed=0)
    ddqn_with_per(agent, env, fname='ddqn_per') 

if __name__ == '__main__':
    main()