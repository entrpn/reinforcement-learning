import numpy as np
from grid_world import standard_grid, negative_grid

# Basically as I understand it, we play many episodes. For each episode we calculate G for each
# state that we passed through in that episode. Then we add that to a global returns = {} dict.
# Then we take the mean value of that return for that state and that gives us a calculated expected 
# value based on its sample mean (sample mean, meaning, a mean calculated across many samples.)


GAMMA = 0.9
ALL_POSSIBLE_ACTIONS = ('U','D','L','R')


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

def play_game(grid, policy, manual = True):

    start_states = list(grid.actions.keys())

    start_idx = np.random.choice(len(start_states))
    # exploring start method. Because MC needs to start at any state.
    grid.set_state(start_states[start_idx])

    if manual:
        print('start_states: ' + str(start_states))
        print('start_idx: ' + str(start_idx))
        print('get_state: ' + str(grid.current_state()))
        print()
        input()

    s = grid.current_state()
    states_and_rewards = [(s,0)]
    while not grid.game_over():
        a = policy[s]
        r = grid.move(a)
        s = grid.current_state()
        states_and_rewards.append((s,r))
        if manual:
            print('action: ' + str(a))
            print('reward: ' + str(r))
            print('state: ' + str(grid.current_state()))
            print('states_and_rewards: ' + str(states_and_rewards))
            print()
            input()
    
    G = 0
    states_and_returns = []
    first = True
    for s, r in reversed(states_and_rewards):
        if first:
            first = False
        else:
            states_and_returns.append((s,G))
        G = r + GAMMA*G
    states_and_returns.reverse() # we want it to be in order of state visited
    if manual:
        print('states_and_returns: ' + str(states_and_returns))
    return states_and_returns

if __name__ == '__main__':
    grid = standard_grid()

    print ('rewards: ')
    print_values(grid.rewards,grid)

    # state -> action
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

    # initialize V(s) and returns
    V = {}
    returns = {} # dictionary of state -> list of returns we've received
    states = grid.all_states()
    for s in states:
        if s in grid.actions:
            returns[s] = []
        else:
            V[s] = 0

    manual = True
    
    for t in range(100):
        states_and_returns = play_game(grid, policy, manual)
        seen_states = set()
        for s, G in states_and_returns:
            # check if we have already seen s
            # called 'first-visit' MC policy evaluation
            if manual:
                print('state: ' + str(s))
                print('return: ' + str(G))
            if s not in seen_states:
                print('not in seen state')
                returns[s].append(G)
                print('returns[s]: ' + str(returns[s]))
                V[s] = np.mean(returns[s])
                print('V[s]: ' + str(V[s]))
                seen_states.add(s)
            elif manual:
                print('already in seen state')
            
            if manual:
                input()
    
    print('values:')
    print_values(V,grid)
    print('policy:')
    print_policy(policy, grid)