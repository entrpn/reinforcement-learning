import gym
# https://github.com/openai/gym/wiki/CartPole-v0
env = gym.make('CartPole-v0')

print(env.reset())

box = env.observation_space
print(box)
print('Action space: ')
print(env.action_space)

# Perform an action
done = False
while not done:
    print('action: ',env.action_space.sample())
    observation, reward, done, _ = env.step(env.action_space.sample())
