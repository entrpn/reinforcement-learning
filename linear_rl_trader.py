import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime
import itertools
import argparse
import re
import os
import pickle

from sklearn.preprocessing import StandardScaler

manual = True
logAll = True

def get_data():
    df = pd.read_csv('./aapl_msi_sbux.csv')
    if logAll:
        print('Data values: ')
        print(df.head())
        print()
    return df.values

def get_scaler(env):

    states = []
    # take random actions and get the state.
    # This creates a distribution because it is not feasable 
    # to get all possible states.
    # For example, one state looks like this:
    # [128.    , 127.    ,   0.    ,  64.2828,  57.29  ,  30.935 , 12.8755]
    # Thats the representation of owning 128 shares of aapl, 127 of msi and 0 of sbux
    # and their respective prices. The final value 12.8755 is the cash left.
    # Imagine how many possibilities there are of this. Is not feasable, so we
    # create a distribution on n number of samples and create the scaler on that data.
    for _ in range(env.n_step):
        action = np.random.choice(env.action_space)
        state, reward, done, info = env.step(action)
        states.append(state)
        if done:
            break

    scaler = StandardScaler()
    scaler.fit(states)
    return scaler

def maybe_make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

class LinearModel:

    def __init__(self, input_dim, n_action):
        self.W = np.random.randn(input_dim,n_action) / np.sqrt(input_dim)
        self.b = np.zeros(n_action)

        self.vW = 0
        self.vb = 0

        self.losses = []

        if logAll:
            print()
            print('//************Creating LinearModel************\\')
            print('W shape: ',self.W.shape)
            print('b shape: ',self.b.shape)
            print('\\**********************************************//')
            print()

    def predict(self, X):
        """X is the state"""
        assert(len(X.shape) == 2)
        return X.dot(self.W) + self.b
    
    def sgd(self, X, Y, learning_rate=0.01, momentum=0.9):
        assert(len(X.shape) == 2)

        # total number of values in Y
        num_values = np.prod(Y.shape)

        # do one step of gradient descent
        # we multiply by 2 to get the exact gradient
        # (not adjusting the learning rate)
        # i.e d/dx (X^2) --> 2x
        Yhat = self.predict(X)
        gW = 2 * X.T.dot(Yhat - Y) / num_values
        gb = 2 * (Yhat - Y).sum(axis=0) / num_values

        self.vW = momentum * self.vW - learning_rate * gW
        self.vb = momentum * self.vb - learning_rate * gb

        #update params
        self.W += self.vW
        self.b += self.vb

        mse = np.mean((Yhat - Y)**2)
        self.losses.append(mse)
    
    def load_weights(self, filepath):
        npz = np.load(filepath)
        self.W = npz['W']
        self.b = npz['b']
    
    def save_weights(self, filepath):
        np.savez(filepath,W=self.W, b=self.b)

class MultiStockEnv:
    """
    A 3-stock trading environment.
    State: vector of size 7 (n_stock * 2 + 1)
        - # shares of stock 1 owned
        - # shares of stock 2 owned
        - # shares of stock 3 owned
        - price of stock 1 owned
        - price of stock 2 owned
        - price of stock 3 owned
        - cash owned (can be used to purchase more stocks)
    Action: categorical variable with 27 (2^3) possibilities
        - for each stock, you can:
        - 0: sell
        - 1: hold
        - 2: buy
    """
    def __init__(self, data, initial_investment=20000):
        self.stock_price_history = data
        self.n_step, self.n_stock = self.stock_price_history.shape

        # instance attributes
        self.initial_investment = initial_investment
        self.cur_step = None
        self.stock_owned = None
        self.stock_price = None
        self.cash_in_hand = None

        # np.arange returns evenly spaced values within a given interval.
        # our action space is 3^(number of stocks(3)) = 27
        self.action_space = np.arange(3**self.n_stock)

        # action permutations
        # returns a nested list with elements like:
        # [0,0,0]
        # [0.0,1]
        # [0,0,2]
        # [0,1,0]
        # [0,1,1]
        # etc.
        # 0 = sell
        # 1 = hold
        # 2 = buy
        self.action_list = list(map(list,itertools.product([0,1,2], repeat=self.n_stock)))

        # calculate size of state
        self.state_dim = self.n_stock * 2 + 1

        self.reset()

        if logAll:
            print()
            print('//************Creating MultiStockEnv************\\')
            print('action space: ',self.action_space)
            print('action list: ',self.action_list)
            print('state size: ',self.state_dim)
            print()
            print('stock owned: ',self.stock_owned)
            print('cash in hand: ',self.initial_investment)
            print('\\**********************************************//')
            print()
    
    def reset(self):
        self.cur_step = 0
        self.stock_owned = np.zeros(self.n_stock)
        self.stock_price = self.stock_price_history[self.cur_step]
        self.cash_in_hand = self.initial_investment
        return self._get_obs()
    
    def step(self, action):
        assert action in self.action_space

        # get current value before performing the action
        prev_val = self._get_val()

        # update price, i.e. go to the next day
        self.cur_step +=1
        self.stock_price = self.stock_price_history[self.cur_step]

        # perform the trade
        self._trade(action)

        cur_val = self._get_val()

        reward = cur_val - prev_val

        done = self.cur_step == self.n_step - 1

        info = {'cur_val': cur_val}

        return self._get_obs(), reward, done, info
    
    def _get_obs(self):
        """This returns an array with number of stock owned, stock price and cash in hand
           Ex: [128, 127, 0, 64.2828, 57.29, 30.935, 12.8755]
        """
        obs = np.empty(self.state_dim)
        obs[:self.n_stock] = self.stock_owned
        obs[self.n_stock:2*self.n_stock] = self.stock_price
        obs[-1] = self.cash_in_hand
        return obs
    
    def _get_val(self):
        """get the value of my portfolio"""
        # dot product of two arrays 
        # Ex:
        # a = np.array([1,2,3])
        # b = np.array([4,5,6])
        # print(a.dot(b)) = 32
        # because 1*4 + 2*5 + 3*6 = 32
        return self.stock_owned.dot(self.stock_price) + self.cash_in_hand

    def _trade(self, action):
        # if logAll:
        #     print('trade(), action: ',action)
        # index the action we want to perform
        # 0 : sell
        # 1 : hold
        # 2 : buy
        # e.g. [2,1,0] means:
        # buy first stock
        # hold second stock
        # sell third stock
        action_vec = self.action_list[action]
        # if logAll:
        #     print('action_list[action]: ',action_vec)

        # determine which stocks to buy or sell
        sell_index = [] # stores index of stocks we want to sell
        buy_index = [] # stores index of stocks we want to buy
        for i, a in enumerate(action_vec):
            if a == 0:
                sell_index.append(i)
            elif a == 2:
                buy_index.append(i)

        # sell any stocks we want to sell
        # then buy any stocks we want to buy
        if sell_index:
            # NOTE: to simplify the problem, when we sell, we will sell ALL shares of that stock
            for i in sell_index:
                self.cash_in_hand += self.stock_price[i] * self.stock_owned[i]
                self.stock_owned[i] = 0
        if buy_index:
            # NOTE: when buying, we will loop through each stock we want to buy,
            # and buy one share at a time until we run out of cash.
            can_buy = True
            while can_buy:
                for i in buy_index:
                    if self.cash_in_hand > self.stock_price[i]:
                        self.stock_owned[i] +=1 # buy one share
                        self.cash_in_hand -= self.stock_price[i]
                    else:
                        can_buy = False

class DQNAgent(object):
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.95 # discount_rate
        self.epsilon = 1.0 # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = LinearModel(state_size, action_size)
        if logAll:
            print()
            print('//************Creating DQNAgent************\\')
            print('state size: ',self.state_size)
            print('action size: ',self.action_size)
            print('discount rate: ',self.gamma)
            print('exploration rate: ',self.epsilon)
            print('\\**********************************************//')
            print()

    def act(self, state):
        if logAll:
            print('act(), state: ',state)
            rand = np.random.rand()
            print('rand: ',rand)
            print('epsilon: ',self.epsilon)
        if rand <= self.epsilon:
            random_choice = np.random.choice(self.action_size)
            print('random choice: ',random_choice)
            if manual:
                input()
            return random_choice
        act_values = self.model.predict(state)
        if logAll:
            print('act_values.shape: ',act_values.shape)
            print('act_values: ',act_values)
            print('argmax: ', np.argmax(act_values[0]))
        if manual:
            input()
        # because our array is is a 1 x 27 np.argmax(act_values[0]) and np.argmax(act_values)
        # gives us the same results.
        return np.argmax(act_values[0])

    def train(self, state, action, reward, next_state, done):
        if done:
            target = reward
        else:
            # TODO - LEFT OFF HERE
            target = reward + self.gamma * np.amax(self.model.predict(next_state), axis=1)
        
        target_full = self.model.predict(state)
        target_full[0,action] = target

        self.model.sgd(state, target_full)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def load(self, name):
        self.model.load_weights(name)
    
    def save(self, name):
        self.model.save_weights(name)
    
def play_one_episode(agent, env, is_train):
    # note: after transforming states are already 1xD
    state = env.reset()
    state = scaler.transform([state])
    done = False

    while not done:
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)
        next_state = scaler.transform([next_state])
        if is_train == 'train':
            agent.train(state, action, reward, next_state, done)
        state = next_state

    return info['cur_val']




if __name__ == '__main__':

    #config
    models_folder = 'linear_rl_trader_models'
    rewards_folder = 'linear_rl_trader_rewards'
    num_episodes = 2000
    batch_size = 32
    initial_investment = 20000

    parser = argparse.ArgumentParser()
    parser.add_argument('-m','--mode', type=str, required=True,help='either "train" or "test"')
    args = parser.parse_args()

    maybe_make_dir(models_folder)
    maybe_make_dir(rewards_folder)

    data = get_data()
    n_timesteps, n_stocks = data.shape

    if logAll:
        print('timesteps from data: ',n_timesteps)
        print('stocks: ',n_stocks)

    n_train = n_timesteps // 2

    if logAll:
        print('train samples: ',n_train)

    train_data = data[:n_train]
    test_data = data[n_train:]

    env = MultiStockEnv(train_data, initial_investment)

    if manual:
        input()

    state_size = env.state_dim
    action_size = len(env.action_space)
    agent = DQNAgent(state_size, action_size)
    if manual:
        input()

    scaler = get_scaler(env)

    # store the final value of the portfolio (end of episode)
    portfolio_value = []

    # if args.mode == 'test':
    #     # then load the previous scaler
    #     with open(f'{models_folder}/scaler.pkl','rb') as f:
    #         scaler = pickle.load(f)

    #     # remake the env with test data
    #     env = MultiStockEnv(test_data, initial_investment)

    #     # make sure epsilon is not 1!
    #     # no need to run multiple episodes if epsilon = 0, it's deterministic
    #     agent.epsilon = 0.01

    #     agent.load(f'{models_folder}/linear.npz')

    # play the game num_episodes times
    for e in range(num_episodes):
        t0 = datetime.now()
        val = play_one_episode(agent, env, args.mode)
    #     dt = datetime.now() - t0
    #     print(f'episode: {e + 1} / {num_episodes}, episode and value: {val:.2f}, duration: {dt}')
    #     portfolio_value.append(val) # append episode end portfolio value
    
    # # save the weights when we are done
    # if args.mode == 'train':
    #     agent.save(f'{models_folder}/linear.npz')

    #     with open(f'{models_folder}/scaler.pkl', 'wb') as f:
    #         pickle.dump(scaler, f)

    #     plt.plot(agent.model.losses)
    #     plt.show()

    # np.save(f'{rewards_folder}/{args.mode}.npy', portfolio_value)

