import numpy as np
import matplotlib.pyplot as plt
from grid_world import standard_grid, negative_grid

manual = False
logall = False

def print_all_Qsa(Q):
    for k,v in Q.items():
        print('state: ',k)
        print('values: ', v)

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

# using epsilon soft
def random_action(a, eps=0.1):
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
ALL_POSSIBLE_ACTIONS = ('U','D','L','R')

if __name__ == '__main__':

    grid = negative_grid(step_cost=-0.1)

    print('rewards: ')
    print_values(grid.rewards,grid)

    Q = {}
    states = grid.all_states()
    for s in states:
        Q[s] = {}
        for a in ALL_POSSIBLE_ACTIONS:
            Q[s][a] = 0
    
    # let's also keep track of how many times Q[s] has been updated
    update_counts = {}
    update_counts_sa = {}
    for s in states:
        update_counts_sa[s] = {}
        for a in ALL_POSSIBLE_ACTIONS:
            update_counts_sa[s][a] = 1.0

    # repeat until convergence
    t = 1.0
    deltas = []
    for it in range(10000):
        if it % 100 == 0:
            t += 10e-3
        if it % 2000 == 0:
            print('it: ' + str(it))
        
        # instead of 'generating' an episode, we will PLAY
        # an episode within this loop
        s = (2,0)
        grid.set_state(s)

        a = max_dict(Q[s])[0]
        a = random_action(a, eps=0.5/t)
        biggest_change = 0
        while not grid.game_over():
            r = grid.move(a)
            s2 = grid.current_state()

            # we need the next action as well since Q(s,a) depends on Q(s',a')
            # if s2 not in policy then it's a terminal state, all Q are 0
            a2 = max_dict(Q[s2])[0]
            a2 = random_action(a2, eps=0.5/t) # epsilon-greedy

            #we will update Q(s,a) AS we experience the episode
            alpha = ALPHA / update_counts_sa[s][a]
            update_counts_sa[s][a] += 0.005
            old_qsa = Q[s][a]
            Q[s][a] = Q[s][a] + alpha*(r + GAMMA*Q[s2][a2] - Q[s][a])
            biggest_change = max(biggest_change, np.abs(old_qsa - Q[s][a]))
            if logall:
                print('s: ',s)
                print('action: ',a)
                print('reward: ',r)
                print('s2: ',s2)
                print('a2: ',a2)
                print('old_qsa: ',old_qsa)
                print('Q[s2][a2]: ',Q[s2][a2])
                print('new Q[s][a]: ',Q[s][a])
            if manual:
                input()
            # we would like to know how olften Q(s) has been updated too
            update_counts[s] = update_counts.get(s,0) + 1

            s = s2
            a = a2
        deltas.append(biggest_change)

    plt.plot(deltas)
    plt.show()            

    policy = {}
    V = {}
    for s in grid.actions.keys():
        a, max_q = max_dict(Q[s])
        policy[s] = a
        V[s] = max_q

    print('update counts:')
    total = np.sum(list(update_counts.values()))
    print(total)
    for k, v in update_counts.items():
        update_counts[k] = float(v) / total
    print_values(update_counts, grid)

    print('values:')
    print_values(V, grid)
    print('policy:')
    print_policy(policy, grid)

    print('all Q[s][a]')
    print_all_Qsa(Q)