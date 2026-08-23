import os
import pickle

import events as e

from .callbacks import ACTIONS, state_to_features


def setup_training(self):
    self.alpha = 0.1
    self.gamma = 0.9


def game_events_occurred(
    self,
    old_game_state: dict,
    self_action: str,
    new_game_state: dict,
    events: list
):
    old_state = state_to_features(old_game_state)
    new_state = state_to_features(new_game_state)

    reward = reward_from_events(events)

    old_q = self.q_table.get((old_state, self_action), 0.0)

    next_q_values = [
        self.q_table.get((new_state, action), 0.0)
        for action in ACTIONS
    ]

    max_next_q = max(next_q_values)

    self.q_table[(old_state, self_action)] = (
        old_q
        + self.alpha * (
            reward
            + self.gamma * max_next_q
            - old_q
        )
    )


def reward_from_events(events: list) -> float:
    reward = -0.1

    if e.COIN_COLLECTED in events:
        reward += 10

    if e.INVALID_ACTION in events:
        reward -= 1

    return reward


def end_of_round(
    self,
    last_game_state: dict,
    last_action: str,
    events: list
):
    last_state = state_to_features(last_game_state)

    reward = reward_from_events(events)

    old_q = self.q_table.get((last_state, last_action), 0.0)

    # Terminal state: es gibt keinen zukünftigen Q-Wert mehr
    self.q_table[(last_state, last_action)] = (
        old_q
        + self.alpha * (
            reward - old_q
        )
    )

    with open("q_table.pkl", "wb") as file:
        pickle.dump(self.q_table, file)