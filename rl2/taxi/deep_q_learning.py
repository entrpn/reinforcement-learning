import numpy as np
import random
from collections import deque

import gym

from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Embedding, Reshape
from tensorflow.keras.optimizers import Adam

import logging

manual = False

logging.basicConfig()
logger = logging.getLogger('deep_q_learning')
logger.setLevel(logging.DEBUG)
#logger.setLevel(logging.INFO)

env = gym.make('Taxi-v3').env
env.render()

print('Number of states: {}'.format(env.observation_space.n))
print('Number of actions {}'.format(env.action_space.n))


class Agent:
    def __init__(self, env, optimizer):

        self._state_size = env.observation_space.n
        self._action_size = env.action_space.n
        self._optimizer = optimizer

        self.experience_replay = deque(maxlen=2000)

        # Initialize discount and exploration rate
        self.gamma = 0.6
        self.epsilon = 0.1

        # Build Networks
        self.q_network = self._build_compile_model()
        self.target_network = self._build_compile_model()
        self.alighn_target_model()

    def store(self, state, action, reward, next_state, done):
        self.experience_replay.append((state, action, reward, next_state, done))

    def alighn_target_model(self):
        self.target_network.set_weights(self.q_network.get_weights())

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            logger.debug('sample action space')
            return env.action_space.sample()
        
        q_values = self.q_network.predict(state)
        logger.debug('q_values: {}'.format(q_values))
        return np.argmax(q_values[0])
    
    def retrain(self, batch_size):
        logger.debug('retrain')
        minibatch = random.sample(self.experience_replay, batch_size)

        for state, action, reward, next_state, done in minibatch:
            target = self.q_network.predict(state)
            if done:
                target[0][action] = reward
            else:
                t = self.target_network.predict(next_state)
                target[0][action] = reward + self.gamma * np.amax(t)

            self.q_network.fit(state, target, epochs=1, verbose=0)

    
    def _build_compile_model(self):
        model = Sequential()
        model.add(Embedding(self._state_size, 10, input_length=1))
        model.add(Reshape((10,)))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(self._action_size, activation='linear'))

        model.compile(loss='mse', optimizer=self._optimizer)
        return model


optimizer = Adam(learning_rate=0.01)
agent = Agent(env, optimizer)

batch_size=32
num_of_episodes=100
timesteps_per_episode = 1000
agent.q_network.summary()

allRewards = []

for e in range(0,num_of_episodes):
    # Reset the environment seems to be initialized at radom every time.
    state = env.reset()
    state = np.reshape(state,[1,1])

    # Initialize variables
    reward = 0
    d = False
    rTotal = reward

    for timestep in range(timesteps_per_episode):
        action = agent.act(state)

        next_state, reward, done, info = env.step(action)
        next_state = np.reshape(next_state,[1,1])
        agent.store(state, action, reward, next_state, done)

        state = next_state

        rTotal += reward

        if manual:
            input()

        if done:
            print('done')
            agent.alighn_target_model()
            break

        if len(agent.experience_replay) > batch_size:
            agent.retrain(batch_size)
    allRewards.append(rTotal)

    if (e+1)%10 == 0:
        print('*********************************')
        print('Episode: {}'.format(e + 1))
        env.render()
        print("*********************************")

print('rewards: ')
print(allRewards)