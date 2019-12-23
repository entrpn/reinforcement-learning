import gym
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from gym import wrappers
from datetime import datetime

# code we already wrote
import q_learning
from q_learning import plot_cost_to_go

 class SGDRegressor:
     def __init__(self, **kwargs):
         self.w = None
         self.lr = 10e-3

    def partial_fit(self,X,Y):
        if self.w is None:
            D = X.shape[1]
            self.w = np.random.randn(D) / np.sqrt(D)
        self.w += self.lr*(Y-X.dot(self.w)).dot(X)

    def predict(self,X):
        return X.dot(self.w)

q_learning.SGDRegressor = SGDRegressor

# calculate everything up to max[Q(s,a)]
# Ex.
# R(t) + gamma*R(t+1) + ... + (gamma^(n-1))*R(t+n-1) + (gamma^n)*max[Q(s(t+n),a(t+n))]

# returns a list of states_and_rewards, and the total reward
def play_one(model, eps, gamma, n=5):
    observation = env.reset()
    done = False
    totalreward = 0;
    rewards = []
    states = []
    actions = []
    iters = 0
    multiplier = np.array([gamma]*n)**np.arange(n)
    print(multiplier)

    while not done and iters < 10000:

        action = model.sample_action(observation, eps)

        states.append(observation)
        actions.append(action)
        
