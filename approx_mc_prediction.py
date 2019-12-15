import numpy as np
import matplotlib.pyplot as plt
from grid_world import standard_grid, negative_grid
from iterative_policy_evaluation_03 import print_policy, print_values

# NOTE: This is only policy evaluation, not optimization

from monte_carlo_random import random_action, play_game, SMALL_ENOUGH, GAMMA

LEARNING_RATE = 0.001

if __name__ == '__main__':

    grid = standard_grid()

    print('rewards:')
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

    # initialize theta
    # our model is V_hat = theta.dot(x)
    # where x = [row, col, row*col, 1] -1 for bias term
    theta = np.random.randn(4) / 2
    print('theta: ',theta)