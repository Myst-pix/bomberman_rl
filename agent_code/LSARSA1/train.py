from collections import namedtuple, deque

import pickle
import numpy as np
from typing import List

import events as e
from .callbacks import action_value_aprox_Function, policy,  action_state_to_features

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
MOVED = "MOVED"
CLOSER_TO_COIN = "CLOSER TO COIN"
MOVED_AWAY_FROM_COIN = "MOVED_AWAY_FROM_COIN"

def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    #learning rate
    self.alpha = 0.001
    #Discount
    self.gamma = 0.5
    #Events


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    old_features = action_state_to_features(old_game_state,self_action)
    check_rewards(old_game_state,new_game_state,self_action,events)
    Reward = reward_from_events(self,events)
    self.action = policy(new_game_state,self)
    next_action = self.action
    TD_error = Reward + self.gamma * action_value_aprox_Function(next_action,new_game_state,self) - action_value_aprox_Function(self_action,old_game_state,self)
    delta_w = self.alpha * TD_error * old_features
    old_model = self.model
    self.model = self.model + delta_w
    print(f'DIFF: {np.linalg.norm(self.model - old_model)}')
    #self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

def check_rewards(old_game_state,new_game_state,self_action,events):
    check_coin_closer(self_action,old_game_state,events)
    check_moved(old_game_state,new_game_state,events)

def check_moved(old_game_state, new_game_state, events):
    if old_game_state['self'][3] != new_game_state['self'][3]:
        events.append(MOVED)
def check_coin_closer(self_action,old_game_state,events):
    new_features = action_state_to_features(old_game_state,self_action)
    if new_features[1] == 1 or e.COIN_COLLECTED in events:
        events.append(CLOSER_TO_COIN)
    else:
        events.append(MOVED_AWAY_FROM_COIN)


def end_of_round(self, last_game_state, last_action, events):
    Reward = reward_from_events(self, events)
    old_features = action_state_to_features(last_game_state, last_action)
    died = e.KILLED_SELF in events or e.GOT_KILLED in events  # adjust to your events.py
    if died:
        TD_error = Reward - action_value_aprox_Function(last_action, last_game_state, self)
    else:
        # truncated, not terminated -> still bootstrap off the final observed state
        next_action = policy(last_game_state, self)
        TD_error = Reward + self.gamma * action_value_aprox_Function(next_action, last_game_state, self) \
                   - action_value_aprox_Function(last_action, last_game_state, self)
    self.model = self.model + self.alpha * TD_error * old_features
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)


def reward_from_events(self, events: List[str]) -> float:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 2,
        e.INVALID_ACTION: -1,
        MOVED_AWAY_FROM_COIN: -1,
        CLOSER_TO_COIN: 1,
        e.WAITED: -1,
        e.SURVIVED_ROUND: 2
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum



