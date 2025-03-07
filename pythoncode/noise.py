# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 11:30:01 2022

@author: Thien Duc Hua
website: thienduchua.github.io
Title: Learning-Based Reconfigurable-Intelligent-Surface-Aided Rate-Splitting Multiple Access Networks
Manuscript: https://ieeexplore.ieee.org/abstract/document/10131984

"""

import numpy as np
class OUActionNoise(object):
    def __init__(self, action_space, mu=0, theta=0.15, max_sigma=0.5, min_sigma=0.05, dt=1e-2, x0=None):
      self.mu           = mu
      self.theta        = theta
      self.sigma        = max_sigma
      self.max_sigma    = max_sigma
      self.min_sigma    = min_sigma
      self.dt           = dt
      self.x0           = x0
      self.action_dim   = action_space.shape[0]
      self.low          = action_space.low
      self.high         = action_space.high
      self.reset()
      
      
    def reset(self):
      #self.state = np.ones(self.action_dim) * self.mu
      self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)
      
    def __call__(self):
      x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.action_dim)
      self.x_prev = x
      return x