import numpy as np
import matplotlib.pyplot as plt
from grid_world import standard_grid, negative_grid

GAMMA = 0.9
ALL_POSSIBLE_ACTIONS = ('U','D','L','R')

# This script implements the monte carlo exploring-starts method for finding optimal policy.

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

def play_game(grid, policy, manual = False):
    if manual:
        input()
    print('**************Start One Game**************')
    start_states = list(grid.actions.keys())
    start_idx = np.random.choice(len(start_states))
    # exploring start method. Because MC needs to start at any state.
    grid.set_state(start_states[start_idx])

    if manual:
        print('start_states: ' + str(start_states))
        print_policy(policy, grid)
        print('start_idx: ' + str(start_idx))
        print('Start State: ' + str(grid.current_state()))
        print()
        input()

    s = grid.current_state()
    a = np.random.choice(ALL_POSSIBLE_ACTIONS)
    print('First random Action: ' + str(a))
    states_actions_rewards = [(s,a,0)]
    seen_states = set()
    seen_states.add(grid.current_state())
    num_steps = 0
    while True:
        r = grid.move(a)
        s = grid.current_state()
        print('Moved to State: ' + str(s) + ' with reward: ' + str(r))
        num_steps +=1
        if s in seen_states:
            # hack so that we don't end up in an infinitely long episode
            # bumping into the wall repeatedly
            reward = -10 / num_steps
            print('State already seen, reward changed to : ' + str(reward))
            states_actions_rewards.append((s,None, reward))
            print('Game is Over')
            break
        elif grid.game_over():
            print('Game is Over')
            states_actions_rewards.append((s,None,r))
            break
        else:
            a = policy[s]
            states_actions_rewards.append((s,a,r))
            print('New Action According to Policy: ' + str(a))
            
        seen_states.add(s)
        print('state: ' + str(grid.current_state()))
        print('states_actions_rewards: ' + str(states_actions_rewards))
        print()
        if manual:
            input()
    G = 0
    states_actions_returns = []
    first = True
    for s, a, r in reversed(states_actions_rewards):
        if first:
            first = False
        else:
            states_actions_returns.append((s,a,G))
        G = r + GAMMA*G
    states_actions_returns.reverse() # we want it to be in order of state visited
    print('states_and_returns: ' + str(states_actions_returns))
    return states_actions_returns

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

if __name__ == '__main__':
    grid = negative_grid(step_cost=-0.9)

    print ('rewards: ')
    print_values(grid.rewards,grid)

    # state -> action
    # initialize a random policy
    policy = {}
    for s in grid.actions.keys():
        policy[s] = np.random.choice(ALL_POSSIBLE_ACTIONS)

    # initialize Q(s,a) and returns
    Q = {}
    returns = {} # dictionary of state -> list of returns we've received
    states = grid.all_states()
    for s in states:
        if s in grid.actions:
            Q[s] = {}
            for a in ALL_POSSIBLE_ACTIONS:
                Q[s][a] = 0 # needs to be initialized to something so we can argmax
                returns[(s,a)] = []
        else:
            pass

    #repeat until convergence
    deltas = []

    manual = False
    
    for t in range(2000):
        if t % 1000 == 0:
            print(t)
        
        # generate an episode using pi
        biggest_change = 0
        states_actions_returns = play_game(grid, policy, manual)
        seen_state_action_pairs = set()
        print('***Policy improvement***')
        for s, a, G in states_actions_returns:
            # check if we have already seen s
            # called 'first-visit' MC policy evaluation
            sa = (s,a)
            print('State: ' + str(s))
            print('Action: ' + str(a))
            if sa not in seen_state_action_pairs:
                old_q = Q[s][a]
                print('old_q: ' + str(Q[s][a]))
                returns[sa].append(G)
                Q[s][a] = np.mean(returns[sa])
                print('new Q[s][a]: ' + str(Q[s][a]))
                biggest_change = max(biggest_change, np.abs(old_q - Q[s][a]))
                print("biggest_change: " + str(biggest_change))
                if manual:
                    input()
                seen_state_action_pairs.add(sa)
            print('Q[s]: ' + str(Q[s]))
        deltas.append(biggest_change)

        #update policy
        for s in policy.keys():
            policy[s] = max_dict(Q[s])[0]
    
    plt.plot(deltas)
    plt.show()

    print('final policy: ')
    print_policy(policy, grid)

    # find V
    V = {}
    for s, Qs in Q.items():
        V[s] = max_dict(Q[s])[1]
    
    print('final values:')
    print_values(V, grid)