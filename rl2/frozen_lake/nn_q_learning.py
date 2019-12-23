import gym
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v0')

# Feed forward network to choose actions
W = tf.Variable(tf.random.uniform([16,4],0,0.01))
def predict(x):
#    print('W: ',W)
    Qout = tf.matmul(x,W)
#    print('Qout: ',Qout)
    return np.argmax(Qout,1), Qout
# inputs1 = tf.keras.Input(shape=(1,16),dtype=tf.float32)
# W = tf.Variable(tf.random.uniform([16,4],0,0.01))
# Qout = tf.matmul(inputs1,W)
# print(Qout.shape)
# predict = tf.argmax(Qout,1)


#nextQ = tf.keras.Input(shape=(1,16),dtype=tf.float32)
# loss = tf.reduce_sum(tf.square(nextQ - Qout))
# trainer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
# updateModel = trainer.minimize(loss)


"""Test"""
# model = tf.keras.Sequential()
# model.add(tf.keras.Input())
"""Test"""

y = 0.99
e = 0.1

num_episodes = 2000

#create list to contain total rewards and steps per episode
jList = []
rList = []

for i in range(num_episodes):
    s = env.reset()
    print(s)
    input()
    rAll = 0
    d = False
    j = 0

    s_identity = tf.eye(16)
    while j < 99:
        j+=1
        # Choose an action by greedily (with e chance of random action) from Q-network
        a, allQ = predict(s_identity[s:s+1])

        if np.random.rand(1) < e:
            a[0] = env.action_space.sample()

        s1,r,d,_ = env.step(a[0])

        _ , Q1 =predict(s_identity[s1:s1+1])

        maxQ1 = np.max(Q1)
        targetQ = allQ 

        targetQ[0,a[0]] = r + y*maxQ1

        input()
        print('s1: ',s1)

