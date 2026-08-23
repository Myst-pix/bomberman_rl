import os
import pickle
import random
from collections import deque
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

    field = game_state["field"]
    _, _, _, (x, y) = game_state["self"]

    free_up = is_free(field, x, y - 1)
    free_right = is_free(field, x + 1, y)
    free_down = is_free(field, x, y + 1)
    free_left = is_free(field, x - 1, y)

    coin_direction = direction_to_nearest_coin(game_state)

    return (
        coin_direction,
        free_up,
        free_right,
        free_down,
        free_left
    )

def is_free(field, x, y):
    if x < 0 or y < 0:
        return False

    if x >= field.shape[0] or y >= field.shape[1]:
        return False

    return field[x, y] == 0

def direction_to_nearest_coin(game_state):
    #Bestimmt mittels Breitensuche den ersten Schritt auf dem kürzesten Weg zum nächsten Coin

    field = game_state["field"]
    _, _, _, start = game_state["self"]
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


def distance_to_nearest_coin(game_state):

    if game_state is None:
        return None

    field = game_state["field"]
    _, _, _, start = game_state["self"]
    coins = set(game_state["coins"])

    if not coins:
        return None

    queue = deque([(start, 0)])
    visited = {start}

    directions = [
        (0, -1),
        (1, 0),
        (0, 1),
        (-1, 0)
    ]

    while queue:
        (x, y), distance = queue.popleft()

        if (x, y) in coins:
            return distance

        for dx, dy in directions:
            nx = x + dx
            ny = y + dy
            next_position = (nx, ny)

            if (
                next_position not in visited
                and is_free(field, nx, ny)
            ):
                visited.add(next_position)
                queue.append((next_position, distance + 1))

    return None