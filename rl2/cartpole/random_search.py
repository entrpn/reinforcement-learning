from __future__ import print_function, division
from builtins import range
# Note: might need to update version of future: pip install -U future

import gym
#from gym import wrappers
import numpy as np
import matplotlib.pyplot as plt

import logging

manual = False


logging.basicConfig()
logger = logging.getLogger('random_search')
#logger.setLevel(logging.DEBUG)
logger.setLevel(logging.INFO)

def get_toggle_action(s, w, actionToggle):
    return 1 if actionToggle else 0

def get_action(s, w):
    retval = s.dot(w)
    logger.debug('\nstate: %s\nw (random array): %s\ns.dot(w): %s',s,w,retval)
    return 1 if retval > 0 else 0

def play_one_episode(env, params):
    observation = env.reset()
    done = False
    t = 0

    actionToggle = False
    while not done and t < 10000:
        #env.render()
        t +=1
        #action = get_toggle_action(observation, params, actionToggle)
        #actionToggle = not actionToggle
        action = get_action(observation, params)
        logger.debug('action: %s',action)
        observation, reward, done, info = env.step(action)
        logger.debug('observation: %s',observation)
        logger.debug('reward: %s',reward)
        logger.debug('done: %s',done)
        logger.debug('info: %s',info)
        if manual:
            input()
        if done:
            break
    return t

def play_multiple_episodes(env, T, params):
    logger.debug('play_multiple_episodes()')
    episode_lengths = np.empty(T)
    logger.debug('episode_lengths size: %s',episode_lengths.shape)

    for i in range(T):
        logger.debug('play episode: %s',i)
        episode_lengths[i] = play_one_episode(env, params)

    avg_length = episode_lengths.mean()
    logger.info('avg length: %s', avg_length)
    return avg_length

def random_search(env):
    episode_lengths = []
    best = 0
    params = None
    for t in range(100):
        new_params = np.random.random(4)*2 - 1
        logger.debug('new_params: %s',new_params)
        avg_length = play_multiple_episodes(env, 100, new_params)
        episode_lengths.append(avg_length)

        if avg_length > best:
            params = new_params
            best = avg_length
    
    return episode_lengths, params

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    episode_lengths, params = random_search(env)
    plt.plot(episode_lengths)
    plt.show()

    # play a final set of episodes
    print('Final run with final weights')
    play_multiple_episodes(env, 100, params)
        