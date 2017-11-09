# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 17:26:17 2017

@author: Jesse
"""

import numpy as np
from utils import policy_iteration

class MDP(object):
    def __init__(self, P, nS, nA, transient, desc=None):
        self.P = P # dictionary of state transition and reward probabilities
        self.nS = nS # number of states
        self.nA = nA # number of actions
        self.transient = transient #List of transient states
        self.desc = desc        
                    
    def step(self, s, a): 
        #Returns next state based on current state, action, 
        #and transition probabilities        
        next_ = self.P[s][a]
        probs = [l[0] for l in next_]        
        ix = np.random.choice(len(next_), size=1, p=probs)[0]
        s, r = next_[ix][1], next_[ix][2]
        return(s, r)
   
    def policyIteration(self, alpha=0.99, nIt=50):
        #Returns sequence of value functions and policies generated for _nIt_ 
        #iterations of policy iteration
        return policy_iteration(self, alpha, nIt)

