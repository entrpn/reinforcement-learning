import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as kl
import gym

class ProbabilityDistribution(tf.keras.Model):
    # When subclassing Model class, we implement the model's forward pass in call
    # https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/keras/Model#__call__
    def call(self, logits):
        return tf.squeeze(tf.random.categorical(logits, 1), axis=-1)

class Model(tf.keras.Model):
    def __init__(self, num_actions):
        super().__init__('mlp_policy')
        self.hidden1 = kl.Dense(128, activation='relu')
        self.hidden2 = kl.Dense(128, activation='relu')
        self.value = kl.Dense(1, name='value')
        # logits are unnormalized log probabilities
        self.logits = kl.Dense(num_actions, name='policy_logits')
        self.dist = ProbabilityDistribution()
    
    # https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/keras/Model#__call__
    def call(self, inputs):
        # inputs is a numpy array, convert to Tensor
        x = tf.convert_to_tensor(inputs, dtype=tf.float32)
        # separate hidden layers from the same input sensor
        hidden_logs = self.hidden1(x)
        hidden_vals = self.hidden2(x)
        return self.logits(hidden_logs), self.value(hidden_vals)
    
    def action_value(self, obs):
        # executes call() under the hood
        logits, value = self.predict(obs)
        print('action_value, logits: ',logits)
        print('action_value, value: ',value)
        action = self.dist.predict(logits)
        # a sompler option, will become clear later why we don't use it
        # action = tf.random.categorical(logits, 1)
        return np.squeeze(action, axis=-1), np.squeeze(value, axis=-1)

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    print('env.action_space.n: ',env.action_space.n)
    model = Model(num_actions=env.action_space.n)

    obs = env.reset()
    print('obs: ',obs)
    print('obs[None,:]',obs[None,:])
    action, value = model.action_value(obs[None, :])
    print('action: ',action)
    print('value: ',value)