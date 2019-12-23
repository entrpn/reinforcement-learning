import gym
import numpy as np

manual = False

env = gym.make('FrozenLake-v0')

print('states: ', env.observation_space.n)
print('actions: ', env.action_space.n)

Q = np.zeros([env.observation_space.n, env.action_space.n])

lr = 0.8
y = .95

num_episodes = 2000
### NOT COMPLETE
# create a list to contain total rewards and steps per episode
rList = []
for i in range(num_episodes):
    s = env.reset()
    rAll = 0
    d = False
    j = 0
    while j < 99:
        j+=1
        rando = np.random.randn(1,env.action_space.n)
        decay = 1/(i+1)
        a = np.argmax(Q[s,:] + rando*decay)
        # Get new state and reward from env
        s1,r,done,_ = env.step(a)
        if manual:
            env.render()
            print('state: ',s)
            print('Q[s,:]: ',Q[s,:])
            print('rando: ',rando)
            print('decay: ',decay)
            print('action: ',a)
            print('s1: ',s1)
            print('reward: ',r)
            print('done: ',done)
            input()
        # Update Q-table with new knowledge
        Q[s,a] = Q[s,a] + lr*(r + y*np.max(Q[s1,:]) - Q[s,a])
        rAll +=r
        s = s1
        if done == True:
            break
    rList.append(rAll)

print('Score over time: ',sum(rList)/num_episodes)
print('Final Q-Table values: ')
print(Q)
