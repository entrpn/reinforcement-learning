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

def random_action(a, eps=0.1):
    # we'll use epsilon-soft to ensure all states are visited
    p = np.random.random()
    if p < (1 - eps):
        return a
    else:
        return np.random.choice(ALL_POSSIBLE_ACTIONS)

def max_dict(d):
    # returns the argmax (key) and max (value) from a dictionary
    # put this into a function since we are using it so often
    max_key = None
    max_val = float('-inf')
    for k, v in d.items():
        if v > max_val:
            max_val = v
            max_key = k
    return max_key, max_val

GAMMA = 0.9
ALPHA = 0.1
ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')

if __name__ == '__main__':

    manual = False

    grid = negative_grid(step_cost=-0.1)

    print('rewards:')
    print_values(grid.rewards, grid)

    # no policy initialization, we will derive our policy from most recent Q

    Q = {}
    states = grid.all_states()
    for s in states:
        Q[s] = {}
        for a in ALL_POSSIBLE_ACTIONS:
            Q[s][a] = 0

    # lets also keep track of how many times Q[s] has been updated
    update_counts = {}
    update_counts_sa = {}
    for s in states:
        update_counts_sa[s] = {}
        for a in ALL_POSSIBLE_ACTIONS:
            update_counts_sa[s][a] = 1.0

    t = 1.0
    deltas = []
    for it in range(10000):
        if it % 100 == 0:
            t += 1e-2
        if it % 2000 == 0:
            print('it: ',it)

        #instead of 'generating' an episode, we will PLAY
        # an episode within this loop
        s = (2,0) # start state
        grid.set_state(s)

        # the first (s, r) tuple is the state we start in and 0
        # (since we don't get a reward) for simply starting the game
        # the last (s, r) tuple is the terminal state and the final reward
        # the value for the terminal state is by definition 0, so we don't
        # care about updating it.
        a, _ = max_dict(Q[s])     
        biggest_change = 0
        if manual: 
            print('a: ' + a)  
            input()
        
        while not grid.game_over():
            a = random_action(a,eps=0.5/t)
            # random action also works, but slower since you can bump into walls
            # a = np.random.choice(ALL_POSSIBzLE_ACTIONS)
            r = grid.move(a)
            s2 = grid.current_state()

            # adaptive learning rate
            alpha = ALPHA / update_counts_sa[s][a]
            update_counts_sa[s][a] += 0.005

            # we will update Q(s,a) AS we experience the episode

            old_qsa = Q[s][a]
            # the difference between SARSA and Q-Learning is with Q-learning
            # we will use this max[a']{Q(s',a')} in our update
            # even if we do not end up taking this action in the next step
            # You can see that even though at the bottom we set a = a2, at
            # the beginning of the while loop, we use eps greedy to possibly change the action.
            a2, max_q_s2a2 = max_dict(Q[s2])
            if manual:
                print('old_qsa: ',old_qsa)
                print('a2: ', a2)
                print('max_q_s2a2: ', max_q_s2a2)
            
            Q[s][a] = Q[s][a] + alpha*(r + GAMMA*max_q_s2a2 - Q[s][a])
            biggest_change = max(biggest_change, np.abs(old_qsa - Q[s][a]))

            update_counts[s] = update_counts.get(s,0) + 1

            s = s2
            a = a2

        deltas.append(biggest_change)

    plt.plot(deltas)
    plt.show()

    #determine the policy from Q*
    # find V* from Q*

    policy = {}
    V = {}
    for s in grid.actions.keys():
        a, max_q = max_dict(Q[s])
        policy[s] = a
        V[s] = max_q
    
    # what's the proportion of time we spend updating each part of Q?
    print('update counts:')
    total = np.sum(list(update_counts.values()))
    for k, v in update_counts.items():
        update_counts[k] = float(v) / total
    print_values(update_counts, grid)

    print('values:')
    print_values(V, grid)
    print('policy:')
    print_policy(policy, grid)
        