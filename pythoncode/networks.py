# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 11:30:01 2022

@author: Thien Duc Hua
website: thienduchua.github.io
Title: Learning-Based Reconfigurable-Intelligent-Surface-Aided Rate-Splitting Multiple Access Networks
Manuscript: https://ieeexplore.ieee.org/abstract/document/10131984

"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


def fan_in_uniform_init(tensor, fan_in=None):
    if fan_in is None:
        fan_in = tensor.size(-1)
        
    weight = 1./np.sqrt(fan_in)
    nn.init.uniform(tensor, -weight, weight)

class CriticNetwork(nn.Module):
    def __init__(self, obs_dims, actions_dims,
                 fc1_dims = 512, fc2_dims=254,
                 # fc1_dims = 1024, fc2_dims=512, #fc3_dims, fc4_dims,
                 init_weight=0.003, init_bias=0.0003
                 #name, chkpt_dir='tmp/ddpg'
                 ):
        super(CriticNetwork, self).__init__()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.fc1 = nn.Linear(obs_dims, fc1_dims)
        self.ln1 = nn.LayerNorm(fc1_dims)
        
        self.fc2 = nn.Linear(fc1_dims + actions_dims, fc2_dims)
        self.ln2 = nn.LayerNorm(fc2_dims)
        
        self.q = nn.Linear(fc2_dims, 1)
        
        fan_in_uniform_init(self.fc1.weight)
        fan_in_uniform_init(self.fc1.bias)
        
        fan_in_uniform_init(self.fc2.weight)
        fan_in_uniform_init(self.fc2.bias)
        
        nn.init.uniform_(self.q.weight, -init_weight, init_weight)
        nn.init.uniform_(self.q.bias, -init_bias, init_bias)
        
    def forward(self, state, action):
        state_value = state
        # layer 1
        state_value = self.fc1(state_value)
        state_value = self.ln1(state_value)
        state_value = F.relu(state_value)
        #layer 2
        state_value = torch.cat((state_value, action), 1)
        state_value = self.fc2(state_value)
        state_value = self.ln2(state_value)
        state_value = F.relu(state_value)

        q_value = self.q(state_value)
        
        return q_value

class ActorNetwork(nn.Module):
    def __init__(self, obs_dims, actions_dims, 
                 fc1_dims = 256, fc2_dims=128,
                 init_weight=0.003, init_bias =0.0003):
        
        super(ActorNetwork, self).__init__()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.fc1 = nn.Linear(obs_dims, fc1_dims)
        # self.linear1 = nn.Linear(obs_dim, h1)
        self.ln1 = nn.LayerNorm(fc1_dims)
        
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.ln2 = nn.LayerNorm(fc2_dims)
        
        self.mu = nn.Linear(fc2_dims, actions_dims)
        
    def forward(self, observation):
        action = self.fc1(observation)
        action = self.ln1(action)
        action = F.relu(action)
        
        action = self.fc2(action)
        action = self.ln2(action)
        action = F.relu(action)
        
        action = torch.sigmoid(self.mu(action))
        
        return action
    
    def get_action(self, observation):
        state = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        action = self.forward(state)
        action = action.squeeze(0).cpu().detach().numpy()
        
        return action

        
        
        
        
        
        