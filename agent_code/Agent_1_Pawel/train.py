from collections import namedtuple, deque

import pickle
import numpy as np
from typing import List

import events as e
from .callbacks import action_value_aprox_Function, policy, nearest_coin, action_state_to_features

CLOSER_TO_COIN = "CLOSER TO COIN"
MOVED = "MOVED"
WAITED = "WAITED"

def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    #learning rate
    self.alpha = 0.1
    #Discount
    self.gamma = 0.9
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
    check_rewards(old_game_state,new_game_state,self_action,events)

    Reward = reward_from_events(self,events)
    old_features = action_state_to_features(old_game_state,self_action)
    #features = state_to_features(new_game_state)
    next_action = policy(new_game_state,self)
    #print(next_action)
    TD_error = Reward + self.gamma * action_value_aprox_Function(next_action,new_game_state,self) - action_value_aprox_Function(self_action,old_game_state,self)
    delta_w = self.alpha * TD_error * old_features
    self.model = self.model + delta_w
    print(self.model)
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

def check_rewards(old_game_state,new_game_state,self_action,events):
    check_coin_closer(old_game_state,new_game_state,events)
    check_moved(self_action,events)

def check_moved(self_action,events):
    match self_action:
        case 'UP':
           events.append(MOVED)
        case 'DOWN':
            events.append(MOVED)
        case 'LEFT':
            events.append(MOVED)
        case 'RIGHT':
            events.append(MOVED)
        case 'WAIT':
            events.append(WAITED)
        case 'BOMB':
            pass
def check_coin_closer(old_game_state,new_game_state,events):
    coin_cord_old = np.array(nearest_coin(old_game_state))
    pos_old = np.array(old_game_state['self'][3])
    pos_new = np.array(new_game_state['self'][3])
    dist_old = np.linalg.norm(coin_cord_old-pos_old)
    dist_new = np.linalg.norm(coin_cord_old-pos_new)
    if dist_new<dist_old:
        events.append(CLOSER_TO_COIN)


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    #todo check this
    # Store the model
    check_moved(last_action, events)
    Reward = reward_from_events(self, events)
    old_features = action_state_to_features(last_game_state, last_action)
    TD_error = Reward - action_value_aprox_Function(last_action, last_game_state, self)
    delta_w = self.alpha * TD_error * old_features
    self.model = self.model + delta_w

    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 2,
        e.INVALID_ACTION: -1,
        e.KILLED_SELF: -5,
        CLOSER_TO_COIN: 1,
        MOVED: -1,
        WAITED: -1
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum



