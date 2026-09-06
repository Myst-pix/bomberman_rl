import os
import pickle
import random
from collections import deque

import numpy as np


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
MOVES = ['UP', 'RIGHT', 'DOWN', 'LEFT']
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
        self.model = np.zeros(3)
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)
    self.action = 'DOWN'



def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """


    if self.train:
        return self.action
    else:
        return policy(game_state,self)

def action_state_to_features(game_state:dict, action) -> np.array:
    if game_state is None:
        return None


    field = game_state["field"]
    _, _, _, (x, y) = game_state["self"]
    features = []
    free = False
    match action:
        case 'UP':
            if is_free(field,x,y-1):
                free = True
        case 'DOWN':
            if is_free(field,x,y+1):
                free = True
        case 'LEFT':
            if is_free(field,x-1,y):
                free = True
        case 'RIGHT':
            if is_free(field,x+1,y):
                free = True
        case 'WAIT':
            pass
        case 'BOMB':
            pass
    features.append(free)
    on_coin = (x, y) in set(game_state["coins"])
    if on_coin:
        features.append(1)
    else:
        coin_direction = direction_to_nearest_coin(game_state, x, y)
        if ACTIONS[coin_direction] == action:
            features.append(1)
        else:
            features.append(0)
    features.append(1)
    return np.array(np.array(features))
def is_free(field, x, y):
    if x < 0 or y < 0:
        return False

    if x >= field.shape[0] or y >= field.shape[1]:
        return False

    return field[x, y] == 0

def direction_to_nearest_coin(game_state,x,y):
    #Bestimmt mittels Breitensuche den ersten Schritt auf dem kürzesten Weg zum nächsten Coin

    field = game_state["field"]
    start = (x,y)
    coins = set(game_state["coins"])

    if not coins:
        return 4

    directions = [
        ((0, -1), 0),
        ((1, 0), 1),
        ((0, 1), 2),
        ((-1, 0), 3)
    ]

    queue = deque()
    visited = {start}

    for (dx, dy), direction_id in directions:
        nx = start[0] + dx
        ny = start[1] + dy

        if is_free(field, nx, ny):
            queue.append(((nx, ny), direction_id))
            visited.add((nx, ny))

    while queue:
        position, first_direction = queue.popleft()
        if position in coins:
            return first_direction
        x, y = position
        for (dx, dy), _ in directions:
            nx = x + dx
            ny = y + dy
            next_position = (nx, ny)

            if (
                    next_position not in visited
                    and is_free(field, nx, ny)
            ):
                visited.add(next_position)
                queue.append((next_position, first_direction))
    return 4

def action_value_aprox_Function(action, game_state:dict, self):
    features = action_state_to_features(game_state,action)
    return features.T @ self.model

def policy(game_state:dict,self):
    random_prob = .1
    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        # 1/6 for any action
        prob = 1/6
        return np.random.choice(ACTIONS, p=[0.2,0.2,0.2,0.2,0.2,0])
    action = 'DOWN'
    actionvalue = 0
    for a in ACTIONS:
        valtemp = action_value_aprox_Function(a,game_state,self)
        if valtemp > actionvalue:
            action = a
            actionvalue = valtemp
    self.logger.debug(f'Querying model for action :. {action}')
    return action