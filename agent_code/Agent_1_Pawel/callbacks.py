import os
import pickle
import random

import numpy as np


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
MAX_DIST = 21.0


def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    if not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        self.model = np.zeros(1)
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)



def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    random_prob = .3
    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        # 1/6 for any action
        prob = 1/6
        return np.random.choice(ACTIONS, p=[0.2,0.2,0.2,0.2,0.2,0])


    self.logger.debug("Querying model for action :.")
    action = policy(game_state,self)
    self.logger.debug(action)
    return action


def state_to_features(game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None
    x, y = game_state['self'][3]
    xCoin, yCoin = nearest_coin(game_state)
    dist_coin = distance(np.array([x,y]),np.array([xCoin,yCoin]))
    # relative offset instead of absolute coords
    return np.array([dist_coin,1])

def action_state_to_features(game_state:dict, action) -> np.array:
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None
    (x,y) = game_state['self'][3]
    (xCoin,yCoin) = nearest_coin(game_state)
    match action:
        case 'UP':
            (x,y) = (x,y-1)
        case 'DOWN':
            (x,y) = (x,y+1)
        case 'LEFT':
            (x,y) = (x-1,y)
        case 'RIGHT':
            (x,y) = (x+1,y)
        case 'WAIT':
            pass
        case 'BOMB':
            pass
    dist_coin = distance(np.array((x,y)),np.array(xCoin,yCoin))
    dist_coin = min(dist_coin, MAX_DIST)/ MAX_DIST
    return np.array([dist_coin])

def distance(p, q):
    return np.linalg.norm(p - q)


def nearest_coin(game_state: dict):
    if len(game_state['coins']) == 0:
        coins = np.array([[0,0]])
    else:
        coins = game_state['coins']
    agent = np.array(game_state['self'][3])
    dist_squared = np.sum((coins - agent) ** 2, axis=1)
    closest_idx = np.argmin(dist_squared)
    closest_coin = coins[closest_idx]
    return closest_coin

def action_value_aprox_Function(action, game_state:dict, self):
    features = action_state_to_features(game_state,action)
    return features.T @ self.model

def policy(game_state:dict,self):
    action_values = [action_value_aprox_Function(a, game_state,self) for a in ACTIONS]
    return ACTIONS[np.argmax(action_values)]