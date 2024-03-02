from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor
import numpy as np
import os
from copy import deepcopy
from evaluate import evaluate_HIV, evaluate_HIV_population
import torch
import torch.nn as nn
import random

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!

class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = capacity # capacity of the buffer
        self.data = []
        self.index = 0 # index of the next cell to be filled
        self.device = device
    def append(self, s, a, r, s_, d):
        if len(self.data) < self.capacity:
            self.data.append(None)
        self.data[self.index] = (s, a, r, s_, d)
        self.index = (self.index + 1) % self.capacity
    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        return list(map(lambda x:torch.Tensor(np.array(x)).to(self.device), list(zip(*batch))))
    def __len__(self):
        return len(self.data)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def greedy_action(network, state):
    device = "cuda" if next(network.parameters()).is_cuda else "cpu"
    with torch.no_grad():
        Q = network(torch.Tensor(state).unsqueeze(0).to(device))
        return torch.argmax(Q).item()


class ProjectAgent:
    def __init__(self, config):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gamma = config['gamma']
        self.batch_size = config['batch_size']
        self.nb_actions = config['nb_actions']
        self.memory = ReplayBuffer(config['buffer_size'], device)
        self.epsilon_max = config['epsilon_max']
        self.epsilon_min = config['epsilon_min']
        self.epsilon_stop = config['epsilon_decay_period']
        self.epsilon_delay = config['epsilon_delay_decay']
        self.epsilon_step = (self.epsilon_max-self.epsilon_min)/self.epsilon_stop
        self.model = self.NeuralNet(config,device) 
        self.nb_grad = config['nb_grad']
        self.criterion = config['criterion']
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['learning_rate'])
        self.target_model = deepcopy(self.model).to(device)
      
    def NeuralNet(self,config,device):
        state_dim = env.observation_space.shape[0]
        nb_action = env.action_space.n 
        nb_neurons = 512
        NN = nn.Sequential(nn.Linear(state_dim, nb_neurons),nn.ReLU(),
                          nn.Linear(nb_neurons, nb_neurons),nn.ReLU(), 
                          nn.Linear(nb_neurons, nb_neurons),nn.ReLU(), 
                          nn.Linear(nb_neurons, nb_neurons),nn.ReLU(),
                          nn.Linear(nb_neurons, nb_neurons),nn.ReLU(),
                          nn.Linear(nb_neurons, nb_action)).to(device)
        return NN
    
    def gradient_step(self):
        if len(self.memory) > self.batch_size:
            X, A, R, Y, D = self.memory.sample(self.batch_size)
            QYmax = self.target_model(Y).max(1)[0].detach()
            update = torch.addcmul(R, 1-D, QYmax, value=self.gamma)
            QXA = self.model(X).gather(1, A.to(torch.long).unsqueeze(1))
            loss = self.criterion(QXA, update.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step() 
    
    def train(self, env, max_episode):
        episode_return = []
        episode = 0
        episode_cum_reward = 0
        state, _ = env.reset()
        epsilon = self.epsilon_max
        step = 0
        prev_val = 0
        update_target = 400
        nb_skip_episodes = 60

        while episode < max_episode:
            # update epsilon
            if step > self.epsilon_delay:
                epsilon = max(self.epsilon_min, epsilon-self.epsilon_step)

            # select epsilon-greedy action
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = greedy_action(self.model, state)

            # step
            next_state, reward, done, trunc, _ = env.step(action)
            self.memory.append(state, action, reward, next_state, done)
            episode_cum_reward += reward

            # train
            for _ in range(self.nb_grad): 
                self.gradient_step()
            if step % update_target == 0: 
                    self.target_model.load_state_dict(self.model.state_dict())

            # next transition
            step += 1
            if done or trunc:
                episode += 1
                if episode > nb_skip_episodes :
                    val_agent = evaluate_HIV(agent=self, nb_episode=1)
                else :
                    val_agent = 0
                
                print("Episode ", '{:3d}'.format(episode), 
                      ", Epsilon ", '{:6.2f}'.format(epsilon), 
                      ", Batch size ", '{:5d}'.format(len(self.memory)), 
                      ", Reward ", '{:.2e}'.format(episode_cum_reward),
                      sep='')
                state, _ = env.reset()
                if val_agent > prev_val:
                    prev_val = val_agent
                    self.best_model = deepcopy(self.model).to(device)
                    path = os.getcwd()
                    self.save(path)
                episode_return.append(episode_cum_reward)
                episode_cum_reward = 0
            else:
                state = next_state
                
        self.model.load_state_dict(self.best_model.state_dict())
        path = os.getcwd()
        self.save(path)
        return episode_return
    
    def act(self, observation, use_random=False):
        if use_random :
            return np.random.randint(0, self.nb_actions)
        else :
            with torch.no_grad():
                Q = self.model(torch.Tensor(observation).unsqueeze(0).to(device))
                return torch.argmax(Q).item()
        
    def save(self,path):
        self.path = path + "/DQN_Agent.pt"
        torch.save(self.model.state_dict(), self.path)
        return
        
    def load(self):
        device = torch.device('cpu')
        self.path = os.getcwd() + "/DQN_Agent.pt"
        self.model = self.NeuralNet({}, device)
        self.model.load_state_dict(torch.load(self.path, map_location=device))
        self.model.eval()
        return

# DQN config
config = {'nb_actions': env.action_space.n,
          'learning_rate': 0.001,
          'gamma': 0.98,
          'buffer_size': 1000000,
          'epsilon_min': 0.02,
          'epsilon_max': 1.,
          'epsilon_decay_period': 19000,
          'epsilon_delay_decay': 100,
          'batch_size': 780,
          'criterion': nn.SmoothL1Loss(),
          'nb_grad': 3}

# Train, save and test the model
#torch.manual_seed(1)
#agent = ProjectAgent(config)
#max_episode = 200
#scores = agent.train(env, max_episode)
#path = os.getcwd()
#agent.save(path)
#agent.load()
#evaluate_HIV(agent=agent, nb_episode=1)
#evaluate_HIV_population(agent=agent, nb_episode=15)
