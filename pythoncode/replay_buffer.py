# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 11:30:01 2022

@author: Thien Duc Hua
website: thienduchua.github.io
Title: Learning-Based Reconfigurable-Intelligent-Surface-Aided Rate-Splitting Multiple Access Networks
Manuscript: https://ieeexplore.ieee.org/abstract/document/10131984

"""

import random
import numpy as np
from collections import deque

class ReplayBuffer(object):
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.num_exp = 0
        self.buffer = deque()
        
    def add(self, s, a, r, s2, done):
        experience = (s, a, r, s2, done)
        if self.num_exp < self.buffer_size:
            self.buffer.append(experience)
            self.num_exp += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)
            
    def size(self):
        return self.buffer_size
    
    def count(self):
        return self.num_exp
    
    def sample(self, batch_size):
        if self.num_exp < batch_size:
            batch = random.sample(self.buffer, self.num_exp)
        else:
            batch = random.sample(self.buffer, batch_size)
        s, a, r, s2, d = map(np.stack, zip(*batch))
        
        return s, a, r, s2, d
    
    def clear(self):
        self.buffer = deque()
        self.num_exp = 0