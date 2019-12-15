### Tic Tac Toe

Learn how to train an agent to play tic tac toe based on greedy epsilon strategy.

### Iterative Policy Evaluation

Learn how to use Bellman's equation to calculate V(s).

### Policy iteration

Learn how to use Bellman's equation to find optimal policy values. 

### Monte Carlo

Basically as I understand it, we play many episodes. For each episode we calculate G for each
state that we passed through in that episode. Then we add that to a global returns = {} dict.
Then we take the mean value of that return for that state and that gives us a calculated expected 
value based on its sample mean (sample mean, meaning, a mean calculated across many samples.)

### Monte Carlo Random

In Monte Carlo above, the transitions were deterministic. 

Here we make them random, so to push the algo to go to the goal state even though there is stocasticity. 

### Monte Carlo Exploring Start

Here we start at a random place with a random action.

