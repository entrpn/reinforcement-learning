import numpy as np
import matplotlib.pyplot as plt
from grid_world import standard_grid

SMALL_ENOUGH = 10e-4 # threshold for convergence

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

if __name__ == '__main__':

### iterative policy ###
    step_through = False

    grid = standard_grid()
    states = grid.all_states()

    V = {}
    for s in states:
        V[s] = 0
    gamma = 1.0

    print('actions: ' + str(grid.actions))
    print('rewards: ' + str(grid.rewards))

    i = 0;
    while True:
        biggest_change = 0
        print_values(V,grid)
        if step_through:
            input()
        for s in states:
            old_v = V[s]
            print('s: ' + str(s))
            print('V[s] : ' + str(old_v))
            if step_through:
                input()

            # V(s) only has value if it's not a terminal state
            if s in grid.actions:

                new_v = 0 # we will accumulate the answer
                p_a = 1.0 / len(grid.actions[s]) # each action has equal probability
                print('probability of each action : ' + str(p_a))
                print('possible actions: ' + str(grid.actions[s]))
                for a in grid.actions[s]:
                    print('move to action: ' + str(a))
                    grid.set_state(s)
                    r = grid.move(a)
                    print(' the reward for action is : ' + str(r))
                    new_v += p_a * (r + gamma * V[grid.current_state()])
                    print('new_v : ' + str(new_v))
                    if step_through:
                        input()
                V[s] = new_v
                print('V[s] is now: ' + str(new_v))
                biggest_change = max(biggest_change, np.abs(old_v - V[s]))
                print('biggest_change: ' + str(biggest_change))
                print_values(V,grid)
        if biggest_change < SMALL_ENOUGH:
            print(i)

            break
    print("values for uniformly random actions:")
    print_values(V,grid)
    print("\n\n")



### fixed policy ###

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
print_policy(policy,grid)

# initialize V(s) = 0
V = {}
for s in states:
    V[s] = 0

gamme = 0.9

while True:
    biggest_change = 0
    for s in states:
        old_v = V[s]

        if s in policy:
            a = policy[s]
            grid.set_state(s)
            r = grid.move(a)
            V[s] = r + gamma * V[grid.current_state()]
            biggest_change = max(biggest_change, np.abs(old_v - V[s]))

    if biggest_change < SMALL_ENOUGH:
        break
print("values for fixed policy: ")
print_values(V,grid)