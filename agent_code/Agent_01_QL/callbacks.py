import os
import pickle
import random

import numpy as np


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT']


def setup(self):
    self.epsilon = 0.1

    if os.path.isfile("q_table.pkl"):
        self.logger.info("Loading Q-table.")
        with open("q_table.pkl", "rb") as file:
            self.q_table = pickle.load(file)
    else:
        self.logger.info("Starting with empty Q-table.")
        self.q_table = {}

def act(self, game_state: dict) -> str:
    state = state_to_features(game_state)

    if self.train and random.random() < self.epsilon:
        return random.choice(ACTIONS)

    q_values = np.array([
        self.q_table.get((state, action), 0.0)
        for action in ACTIONS
    ])

    best_actions = np.flatnonzero(q_values == q_values.max())

    action_index = np.random.choice(best_actions)

    return ACTIONS[action_index]

def state_to_features(game_state: dict):
    if game_state is None:
        return None

    _, _, _, (x, y) = game_state["self"]

    coins = game_state["coins"]

    if len(coins) == 0:
        return (x, y, -1, -1)


    #Manhatten Distanz
    nearest_coin = min(
        coins,
        key=lambda coin: abs(coin[0] - x) + abs(coin[1] - y)
    )
    coin_x, coin_y = nearest_coin


    return (x, y, coin_x, coin_y)