import numpy as np
import tensorflow as tf

from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import SGD

tf.compat.v1.enable_eager_execution()

print(tf.__version__)
# Last bandit is set to return the most positive reward
bandits = [0.2,0,-0.2,-5]
num_bandits = len(bandits)
EPS = 0.2

manual = False

def pull(bandit):
    result = np.random.randn(1)
    if result > bandit:
        return 1
    else:
        return -1

def loss(responsible_weight,reward):
    return -(tf.math.log(responsible_weight)*reward)

def train(agent,action,reward, learning_rate=0.001):
    with tf.GradientTape() as t:
        current_loss = loss(agent(action),reward)
    dW = t.gradient(current_loss,[agent.weights])
    weights_as_np = agent.weights.numpy()
    responsible_weight = agent.weights[action]
    responsible_weight_dw = np.array(dW)[0][action]

    weights_as_np[action] = weights_as_np[action] - learning_rate*responsible_weight_dw

    agent.weights.assign(tf.Variable(weights_as_np))

class Agent:
    def __init__(self):
        self.weights = tf.Variable(tf.ones([num_bandits]))

    def __call__(self,action):
        return self.weights[action]

    def act(self):
        if np.random.rand(1) < EPS:
            return np.random.randint(num_bandits)
        else:
            return tf.argmax(self.weights,0).numpy()

total_episodes = 100
total_reward = np.zeros(num_bandits)
e = 0.1
i = 0

agent = Agent()

Ws = []
while i < total_episodes:
    i+=1
    Ws.append(agent.weights.numpy())
    action = agent.act()
    reward = pull(bandits[action])
    if manual:
        print('reward: ', reward)
        print('action: ',action)
        print('weights: ',agent.weights.numpy())
        input()

    print('epoch {}, weights: {}, rewards: {}'.format(i,agent.weights.numpy(),total_reward))
    #Update network
    train(agent,action,reward)

    total_reward[action] += reward
