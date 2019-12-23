from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

TRUE_W = 3.0
TRUE_b = 2.0
NUM_SAMPLES = 1000

def loss(predicted_y, target_y):
    return tf.reduce_mean(tf.square(predicted_y-target_y))

def train(model, inputs, outputs, learning_rate):
    with tf.GradientTape() as t:
        current_loss = loss(model(inputs),outputs)
    dW, db = t.gradient(current_loss,[model.W, model.b])
    model.W.assign_sub(learning_rate * dW)
    model.b.assign_sub(learning_rate * db)

class Model(object):
    def __init__(self):
        self.W = tf.Variable(5.0)
        self.b = tf.Variable(0.0)

    def __call__(self,x):
        return self.W * x + self.b

model = Model()
assert model(3.0).numpy() == 15.0

inputs = tf.random.normal(shape=[NUM_SAMPLES])
noise = tf.random.normal(shape=[NUM_SAMPLES])
outputs = inputs * TRUE_W + TRUE_b + noise

model_outputs = model(inputs)

plt.scatter(inputs, outputs, c='b')
plt.scatter(inputs,model_outputs,c='r')
plt.show()

print('Current loss: %1.6f'%loss(model_outputs,outputs))

Ws,bs = [], []
epochs = range(10)
for epoch in epochs:
    Ws.append(model.W.numpy())
    bs.append(model.b.numpy())
    current_loss = loss(model(inputs), outputs)

    train(model, inputs, outputs, learning_rate=0.1)
    print('Epoch %2d: W=%1.2f b=%1.2f, loss=%2.5f' %(epoch, Ws[-1], bs[-1], current_loss))

plt.plot(epochs,Ws,'r')
plt.plot(epochs,bs,'b')
plt.plot([TRUE_W]*len(epochs),'r--')
plt.plot([TRUE_b]*len(epochs), 'b--')
plt.legend(['W','b','True W', 'True b'])
plt.show()