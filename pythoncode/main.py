# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 11:30:01 2022

@author: Thien Duc Hua
website: thienduchua.github.io
Title: Learning-Based Reconfigurable-Intelligent-Surface-Aided Rate-Splitting Multiple Access Networks
Manuscript: https://ieeexplore.ieee.org/abstract/document/10131984

"""

import torch
import torch.optim as opt
import torch.nn as nn
import torch.nn.functional as F
import scipy.io as sio
from scipy.special import softmax
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from env_rsma_irs import Environment
from matplot import subplot
from noise import OUActionNoise
from networks import CriticNetwork, ActorNetwork
from replay_buffer import ReplayBuffer


############################# GENERAL SETTINGS ###########################################
max_episode = 10000 
max_step = 500     
batch_size = 32         
gamma = 0.9          
tau = 0.001             
buffer_maxlen = 100000   
lr_critic = 0.0001       
lr_actor = 0.0001    
epsilon = 1
epsilon_decay = 0.001 
buffer_start = 100
print_every = 10
###########################################################################################

################################ SETUP TRAINING ###########################################
envrm = Environment()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

state_dim = envrm.observation_space.shape[0]
action_dim = envrm.action_space.shape[0]
NUM_ANTENNA = envrm.NUM_ANTENNA
NUM_ELEMENT = envrm.NUM_ELEMENT
NUM_USER = envrm.NUM_USER

print('State_dim: {}, Action_dim: {}, Number of BS antenna: {}, Number of IRS element: {}, Number of User: {}'.format(state_dim, action_dim, NUM_ANTENNA, NUM_ELEMENT, NUM_USER))
action_noise = OUActionNoise(envrm.action_space)

critic = CriticNetwork(state_dim, action_dim).to(device)
actor = ActorNetwork(state_dim, action_dim). to(device)

target_critic = CriticNetwork(state_dim, action_dim).to(device)
target_actor = ActorNetwork(state_dim, action_dim). to(device)

for target_param, param in zip(target_critic.parameters(), critic.parameters()):
    target_param.data.copy_(param.data)
for target_param, param in zip(target_actor.parameters(), actor.parameters()):
    target_param.data.copy_(param.data)
    
q_optimizer = opt.Adam(critic.parameters(), lr=lr_critic)
policy_optimizer = opt.Adam(actor.parameters(), lr=lr_actor)

MSE = nn.MSELoss()

memory = ReplayBuffer(buffer_maxlen)
###########################################################################################

############################### ITERATE THROUGH EPISODES ##################################
plot_reward = []
plot_policy = []
plot_q = []
plot_steps = []
plot_sum_rate = []

best_reward = -np.inf
saved_reward = -np.inf
saved_ep = 0
average_reward = 0
global_step = 0

def SigmoidFunction(x):
    z = 1/(1 + np.exp(-x))
    return z

def process_Action(action, state):
    actionOut = action
    
    active_bf_part = actionOut[: 2*NUM_ANTENNA*NUM_USER]
    alpha = actionOut[2*NUM_ANTENNA*NUM_USER : 2*NUM_ANTENNA*NUM_USER + NUM_USER]
    delta = actionOut[2*NUM_ANTENNA*NUM_USER + NUM_USER :]
    
    F_mapping = active_bf_part.reshape((2*NUM_ANTENNA, NUM_USER), order='F')
    active_bf_soft = np.zeros_like(F_mapping)
    for i in range(NUM_USER):
        active_bf_soft[:,i] = softmax(F_mapping[:,i])
    
    active_bf = (active_bf_soft[::2] + active_bf_soft[1::2]*1j).reshape((NUM_ANTENNA, NUM_USER), order='F')
    return actionOut, active_bf, alpha, delta

for episode in range(max_episode):
    
    s = deepcopy(envrm.reset())
    
    ep_reward = 0.
    ep_q_value = 0.
    step = 0
    epsilon -= epsilon_decay
    for step in range(max_step):
        global_step += 1
        
        a_opt = actor.get_action(s)
        a_opt = a_opt + action_noise()*max(0, epsilon)
        a_opt = SigmoidFunction(a_opt)
        a_opt, active_bf, alpha, delta = process_Action(a_opt, s)
                
        reward, s2, terminal = envrm.step(s, global_step, active_bf, alpha, delta)
        
        memory.add(s, a_opt, reward, s2, terminal)
        ep_reward += reward
        
        s = deepcopy(s2)
        
        if memory.count() > buffer_start:
            s_batch, a_batch, r_batch, s2_batch, t_batch = memory.sample(batch_size)
            s_batch = torch.FloatTensor(s_batch).to(device)
            a_batch = torch.FloatTensor(a_batch).to(device)
            r_batch = torch.FloatTensor(r_batch).unsqueeze(1).to(device)
            s2_batch = torch.FloatTensor(s2_batch).to(device)
            t_batch = torch.FloatTensor(np.float32(t_batch)).unsqueeze(1).to(device)
            
            a2_batch = target_actor(s2_batch).to(device) # a(t+1) of target_actor 
            
            target_q = target_critic(s2_batch, a2_batch).to(device) 
            y = r_batch + gamma*target_q.detach()
            q = critic(s_batch, a_batch) 
            
            q_optimizer.zero_grad()
            q_loss = MSE(q, y) 
            q_loss.backward()
            q_optimizer.step()
            
            policy_optimizer.zero_grad()
            policy_loss = -critic(s_batch, actor(s_batch)).to(device)
            policy_loss = policy_loss.mean().to(device)
            policy_loss.backward()
            policy_optimizer.step()
            
            for target_param, param in zip(target_critic.parameters(), critic.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
            for target_param, param in zip(target_actor.parameters(), actor.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    plot_reward.append([ep_reward, episode+1])
    average_reward += ep_reward
    
    if ep_reward > best_reward:
        torch.save(actor.state_dict(), 'best_model.pkl')
        best_reward = ep_reward
        saved_reward = ep_reward
        saved_ep = episode+1
        
    if (episode % print_every) == (print_every-1):
        np.savetxt('train_reward.csv', plot_reward, delimiter=',')
        