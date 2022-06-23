# AI for self driving car 

# Importing the libraries 

import numpy as np
import random 
import os 
import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
import torch.autograd as autograd 
from torch.autograd import Variable 

# Creating the architecture of the Neural Network 

class Network(nn.Module):
    
    def __init__(self, input_size, nb_action):
        super(Network, self).__init__()
        self.input_size = input_size 
        self.nb_action = nb_action 
        self.fc1 = nn.Linear(input_size, 30) 
        self.fc2 = nn.Linear(30, nb_action) 
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        q_values = self.fc2(x)
        return q_values 
    
# Implementing Experience Replay 
    
class ReplayMemory(object):
    
    def __init__(self, capacity):
        self.capacity = capacity 
        self.memory = []
        
    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]
            
    def sample(self, batch_size):
        samples = zip(*random.sample(self.memory, batch_size))
        return map(lambda x: Variable(torch.cat(x, 0)), samples)
    
#   Implementing Deep Q Learning 

class Dqn():
    
    def __init__(self, input_size, nb_action, gamma):
        self.gamma = gamma 
        self.reward_window = []
        self.model = Network(input_size, nb_action) 
        self.memory = ReplayMemory(100000) 
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.001)
        self.last_state = torch.Tensor(input_size).unsqueeze(0) 
        self.last_action = 0 
        self.last_reward = 0 
        
        