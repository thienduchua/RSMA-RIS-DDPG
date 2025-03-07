# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 11:30:01 2022

@author: Thien Duc Hua
website: thienduchua.github.io
Title: Learning-Based Reconfigurable-Intelligent-Surface-Aided Rate-Splitting Multiple Access Networks
Manuscript: https://ieeexplore.ieee.org/abstract/document/10131984

"""

import numpy as np
from argparse import ArgumentParser
import math
import cmath
import gym
from gym import spaces
from gym.utils import seeding
from scipy.spatial import distance
from scipy.fftpack import fft
import matplotlib.pyplot as plt
import matplotlib.patches as pltp

'''
Hyperparameters of the environment
'''

parser = ArgumentParser(description = 'Paper Simulation Settings')

parser.add_argument('--SEED_VALUE', default= 7777777,
                    help='The value of sedd value for random initialization')
# number of user, BS antenna, IRS element
parser.add_argument('--NUM_ANTENNA', default=10,
                    help='The number of BS antenna')
parser.add_argument('--NUM_USER', default= 70,
                    help='The number of mobile users')
parser.add_argument('--NUM_ELEMENT', default= 100,
                    help='The number of IRS element')

parser.add_argument('--BANDWIDTH_BS', default= 1000000, 
                    help='The bandwidth at the BS (Hz)')
parser.add_argument('--NOISE_VARIANCE', default= 398.1* (10**(-12)) ,
                    help='The noise variance at the receiver - calculate SINR - sigma squared (mW)')
parser.add_argument('--NOISE_DENSITY', default= 3, 
                    help='The noise power spectral density - N_0 (mW)')
parser.add_argument('--Nx', default= 10,
                    help='IRS element for get-index-IRS equation ')
parser.add_argument('--RICIAN_FACTOR', default= 10, 
                    help='Rician Factor for calculating path loss ')
parser.add_argument('--SCALE_FACTOR', default= 100, 
                    help='Dont know what that is but Jiang paper have it, can be viewed as (downlink noise_power_dB- downlink Pt) ')

hyper_params = parser.parse_args()

class Environment (gym.Env):
    
    NUM_USER = hyper_params.NUM_USER
    NUM_ANTENNA = hyper_params.NUM_ANTENNA
    NUM_ELEMENT = hyper_params.NUM_ELEMENT
    
    indexState_G = np.zeros([NUM_ANTENNA, NUM_ELEMENT], dtype = int)
    indexState_h = np.zeros([NUM_ELEMENT, NUM_USER], dtype = int)
    
    
    
    Nx = hyper_params.Nx
    BANDWIDTH = hyper_params.BANDWIDTH_BS
    NOISE = hyper_params.NOISE_VARIANCE
    RICIAN_FACTOR = hyper_params.RICIAN_FACTOR
    SCALE_FACTOR = hyper_params.SCALE_FACTOR
    
    location_BS = np.array([100, -100, 0])
    location_IRS = np.array([0,0,0])
    
    POWER_TX = np.zeros([NUM_USER,], dtype = float)
    
    location_USER = np.empty([NUM_USER, 3])
    
    NUMBER_STATE = (NUM_ANTENNA * NUM_ELEMENT) + (NUM_ELEMENT * NUM_USER) + NUM_USER
    NUMBER_ACTION = (2*NUM_ANTENNA * NUM_USER) + NUM_USER + NUM_ELEMENT

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def __init__(self,):
        self.seed()
        
        self.generate_txpower()
        self.generate_location()
        
        self.action_space = spaces.Box(low = 0.0, high = 1.0, shape=(self.NUMBER_ACTION,), dtype = float)
        self.observation_space = spaces.Box(low = np.inf, high = np.inf, shape=(self.NUMBER_STATE,), dtype = complex)
    

    def generate_txpower(self):
        POWER_TX_dbm = np.full((self.NUM_USER),15)
        POWER_TX = np.zeros([self.NUM_USER,], dtype=float)
        for index_user in range(self.NUM_USER):
            POWER_TX[index_user] = 10**(POWER_TX_dbm[index_user] / 10)
        
        return POWER_TX
    
    def generate_location(self):  
        for index_user in range(self.NUM_USER):
            x = np.random.uniform(5,35)
            y = np.random.uniform(-35,35)
            z = -15
            coordinate_k = np.array([x, y, z])
            self.location_USER[index_user, :] = coordinate_k
        return self.location_USER
    
    def calculate_pathloss(self):
        self.location_USER = self.generate_location()
        
        d0 = np.linalg.norm(self.location_BS - self.location_IRS)
        pathloss_irs_bs = 30 + 22*np.log10(d0)
        aoa_bs = (self.location_IRS[0] - self.location_BS[0]) / d0
        aod_irs_y = (self.location_IRS[1] - self.location_BS[1]) / d0
        aod_irs_z = (self.location_BS[2] - self.location_IRS[2]) / d0
        
        pathloss_irs_user = []
        aoa_irs_y = []
        aoa_irs_z = []
        for index_user in range(self.NUM_USER):
            d_k = np.linalg.norm(self.location_USER[index_user] - self.location_IRS)
            pathloss_irs_user.append(30 + 22*np.log10(d_k))
            aoa_irs_y_k = (self.location_USER[index_user][1] - self.location_IRS[1]) / d_k
            aoa_irs_z_k = (self.location_USER[index_user][2] - self.location_IRS[2]) / d_k
            
            aoa_irs_y.append(aoa_irs_y_k)
            aoa_irs_z.append(aoa_irs_z_k)
        
        aoa_irs_y = np.array(aoa_irs_y)
        aoa_irs_z = np.array(aoa_irs_z)
        pathloss_irs_user = np.array(pathloss_irs_user)
        
        return pathloss_irs_bs, pathloss_irs_user, aoa_bs, aod_irs_y, aod_irs_z, aoa_irs_y, aoa_irs_z
    
    def generate_channel(self):
        channel_bs_irs = []
        channel_irs_user = []
        
        pathloss_irs_bs, pathloss_irs_user, aoa_bs, aod_irs_y, aod_irs_z, aoa_irs_y, aoa_irs_z = self.calculate_pathloss()
        
        pathloss_irs_bs = pathloss_irs_bs - self.SCALE_FACTOR / 2
        pathloss_irs_bs = np.sqrt(10**((-pathloss_irs_bs)/10))
        
        pathloss_irs_user = pathloss_irs_user - self.SCALE_FACTOR / 2
        pathloss_irs_user = np.sqrt(10**((-pathloss_irs_user)/10))
        
        tmp = np.random.normal(loc=0, scale=np.sqrt(0.5), size=[self.NUM_ANTENNA, self.NUM_ELEMENT]) \
            + 1j * np.random.normal(loc=0, scale=np.sqrt(0.5), size=[self.NUM_ANTENNA, self.NUM_ELEMENT])
        a_bs = np.exp(1j * np.pi * aoa_bs * np.arange(self.NUM_ANTENNA))
        a_bs = np.reshape(a_bs, [self.NUM_ANTENNA, 1])
        
        i1 = np.mod(np.arange(self.NUM_ELEMENT), self.Nx)
        i2 = np.floor(np.arange(self.NUM_ELEMENT) / self.Nx)
        
        a_irs_bs = np.exp(1j * np.pi * (i1*aod_irs_y + i2*aod_irs_z))
        a_irs_bs = np.reshape(a_irs_bs, [self.NUM_ELEMENT, 1])
        
        los_irs_bs = a_bs @ np.transpose(a_irs_bs.conjugate())
        
        tmp = np.sqrt(self.RICIAN_FACTOR / (1 + self.RICIAN_FACTOR)) * los_irs_bs + \
            np.sqrt(1 / (1 + self.RICIAN_FACTOR)) * tmp
        
        tmp = tmp * pathloss_irs_bs
        channel_bs_irs.append(tmp)
        channel_bs_irs = np.array(channel_bs_irs)
        
        tmp = np.random.normal(loc=0, scale=np.sqrt(0.5), size=[self.NUM_ELEMENT, self.NUM_USER]) \
            + 1j * np.random.normal(loc=0, scale=np.sqrt(0.5), size=[self.NUM_ELEMENT, self.NUM_USER])
        for k in range(self.NUM_USER):
            a_irs_user = np.exp(1j * np.pi * (i1*aoa_irs_y[k] + i2*aoa_irs_z[k]))
            tmp[:, k] = np.sqrt(self.RICIAN_FACTOR / (self.RICIAN_FACTOR + 1))*a_irs_user + \
                np.sqrt(1 / (self.RICIAN_FACTOR + 1)) * tmp[:, k]
            tmp[:, k] = tmp[:, k] * pathloss_irs_user[k]
        channel_irs_user.append(tmp)
        channel_irs_user = np.array(channel_irs_user)

        return channel_bs_irs, channel_irs_user
    
        
    def formulate_new_observation(self):
        state = np.zeros([self.NUMBER_STATE,], dtype = complex)
        
        channel_bs_irs, channel_irs_user = self.generate_channel()
        channel_bs_irs_flatten = channel_bs_irs.flatten('F')
        channel_irs_user_flatten = channel_irs_user.flatten('F')
        power_tx = self.generate_txpower()
        
        state = np.concatenate((channel_bs_irs_flatten, channel_irs_user_flatten, power_tx))
        
        return state

    def reset(self):
        state = self.formulate_new_observation()
        return state
    
    def next_observation(self):
        return self.formulate_new_observation()

    def step(self, state, global_step, active_bf, alpha, delta):
        step_reward = 0.0
        channel_bs_irs_part = state[ : self.NUM_ANTENNA*self.NUM_ELEMENT]
        channel_irs_user_part = state[self.NUM_ANTENNA*self.NUM_ELEMENT : self.NUM_ANTENNA*self.NUM_ELEMENT + self.NUM_ELEMENT * self.NUM_USER]
        power_tx = state[self.NUM_ANTENNA*self.NUM_ELEMENT + self.NUM_ELEMENT * self.NUM_USER : self.NUM_ANTENNA*self.NUM_ELEMENT + self.NUM_ELEMENT * self.NUM_USER + self.NUM_USER]
        
        channel_bs_irs = np.reshape(channel_bs_irs_part, (self.NUM_ANTENNA, self.NUM_ELEMENT), order='F')
        channel_irs_user = np.reshape(channel_irs_user_part, (self.NUM_ELEMENT, self.NUM_USER), order='F')

        tx_subpower = np.zeros([self.NUM_USER,2], dtype = float)
        for k in range(self.NUM_USER):
            tx_subpower[k][0] = alpha[k] * power_tx[k]
            tx_subpower[k][1] = (1-alpha[k]) * power_tx[k]

            
        theta = np.zeros([self.NUM_ELEMENT, self.NUM_ELEMENT], dtype=complex)
        for index_vertical in range(self.NUM_ELEMENT):
            for index_horizon in range(self.NUM_ELEMENT):
                if index_vertical == index_horizon:
                    theta[index_vertical, index_horizon] = np.exp(1j*2*np.pi*delta[index_vertical])
        
        ThetaH = np.dot(theta, channel_irs_user) 
        GThetaH = np.dot(channel_bs_irs, ThetaH) 
        active_bf_transpose = np.transpose(active_bf)
        
        activeBF_channel = (np.abs(np.dot(active_bf_transpose, GThetaH)))**2
        
        activeBF_channel_subpower = np.dot(activeBF_channel, tx_subpower)

        activeBF_channel_subpower_flat = np.sort(activeBF_channel_subpower.flatten())[::-1]
        sinr = np.zeros_like(activeBF_channel_subpower_flat)
        for i in range(len(activeBF_channel_subpower_flat)-1):
            sinr[i] = activeBF_channel_subpower_flat[i] / (sum(activeBF_channel_subpower_flat[i+1: ]) + self.NOISE)
        sinr[len(activeBF_channel_subpower_flat)-1] = activeBF_channel_subpower_flat[len(activeBF_channel_subpower_flat)-1] / self.NOISE
        
        self.rate = np.zeros_like(sinr)
        for i in range(len(self.rate)-1):
            self.rate[i] = np.log2(1 + sinr[i])
                        
        step_reward = step_reward + np.sum(self.rate)
        
        observation_next = self.next_observation()
        
        done = False 
        
        return step_reward, observation_next, done
    
