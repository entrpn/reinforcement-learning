import numpy as np
import matplotlib.pyplot as plt
from grid_world import standard_grid, negative_grid

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

SMALL_ENOUGH = 10e-4
GAMMA = 0.9
ALL_POSSIBLE_ACTIONS = ('U','D','L','R')

# next state and reward will now have some randomness
# you'll go in your desired direction with probabiliyt 0.5
# you'll go in a random direction a' != a with probability 0.5/3

if __name__ == '__main__':
    # this grid gives you a reward of -0.1 for every non-terminal state
    # we want to see if this will encourage finding a shorter path to the goal
    grid = negative_grid(step_cost=-1.0)

    print('rewards:')
    print_values(grid.rewards,grid)

    # state -> action
    # we'll randomly choose an action and update as we learn
    policy = {}
    for s in grid.actions.keys():
        policy[s] = np.random.choice(ALL_POSSIBLE_ACTIONS)
    
    #initial policy
    print('initial policy')
    print_policy(policy,grid)

    # initialize V(s)
    V = {}
    states = grid.all_states()
    for s in states:
        if s in grid.actions:
            V[s] = np.random.random()
        else:
            # terminal state
            V[s] = 0

    # repeat until convergence
    while True:

        # policy evaluation step
        while True:
            biggest_change = 0
            for s in states:
                old_v = V[s]

                # V(s) only has value if it's not a terminal state
                new_v = 0
                if s in policy:
                    for a in ALL_POSSIBLE_ACTIONS:
                        if a == policy[s]:
                            p = 0.5
                        else:
                            p = 0.5/3
                        grid.set_state(s)
                        r = grid.move(a)
                        new_v += p*(r + GAMMA*V[grid.current_state()])
                    V[s] = new_v
                    biggest_change = max(biggest_change, np.abs(old_v - V[s]))
            if biggest_change < SMALL_ENOUGH:
                break
        
        # policy improvement step
        is_policy_converged = True
        for s in states:
            if s in policy:
                old_a = policy[s]
                new_a = None
                best_value = float('')
                ### I didn't finish this. Moved to Monte Carlo.