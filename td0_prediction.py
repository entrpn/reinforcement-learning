import numpy as np
import matplotlib.pyplot as plt
from grid_world import standard_grid, negative_grid

manual = False

def print_values(V,g):
    for i in range(g.width):
        print("----------------------------")
        for j in range(g.height):
            v = V.get((i,j),0)
            if v >=0:
                print(' %.2f|' % v, end=" ")
            else:
                print('%.2f|'%v, end= " ")
        print ("")

def print_policy(P,g):
    for i in range(g.width):
        print("----------------------------")
        for j in range(g.height):
            a = P.get((i,j),' ')
            print(' %s |' %a, end = ' ')
        print()

SMALL_ENOUGH = 10-4
GAMMA = 0.9
ALPHA = 0.1
ALL_POSSIBLE_ACTIONS = ('U','D','L','R')

# NOTE: this is only policy evaluation, not optimization

def random_action(a, eps=0.1):
    # we'll use epsilon-soft to ensure all states are visited
    p = np.random.random()
    if p < (1 - eps):
        return a
    else:
        return np.random.choice(ALL_POSSIBLE_ACTIONS)

def play_game(grid, policy):
    s = (2, 0)
    grid.set_state(s)
    states_and_rewards = [(s,0)]
    while not grid.game_over():
        a = policy[s]
        # explore/exploit here
        a = random_action(a)
        r = grid.move(a)
        s = grid.current_state()
        states_and_rewards.append((s,r))
    return states_and_rewards

if __name__ == '__main__':
    
    grid = standard_grid()

    print('rewards: ')
    print_values(grid.rewards, grid)

    policy = {
        (2,0): 'U',
        (1,0): 'U',
        (0,0): 'R',
        (0,1): 'R',
        (0,2): 'R',
        (1,2): 'R',
        (2,1): 'R',
        (2,2): 'R',
        (2,3): 'U',
    }

    V = {}
    states = grid.all_states()
    for s in states:
        V[s] = 0

    for it in range(1000):

        states_and_rewards = play_game(grid, policy)

        print('states_and_rewards size: ',len(states_and_rewards))
        for t in range(len(states_and_rewards) -1):
            s, _ = states_and_rewards[t]
            s2, r = states_and_rewards[t+1]
            print('V[s]: ',V[s])
            V[s] = V[s] + ALPHA*(r + GAMMA*V[s2] - V[s])
            print('s: ',s)
            print('s2: ',s2)
            print('r: ',r)
            print('V[s2]: ',V[s2])
            print('V[s] (after): ',V[s])

            if manual:
                input()
        
    
    print('values:')
    print_values(V, grid)
    print('policy:')
    print_policy(policy, grid)
