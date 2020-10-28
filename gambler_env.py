#!/usr/bin/env python

"""
  Author: Adam White, Mohammad M. Ajallooeian, Sina Ghiassian
  Purpose: Code for the Gambler's problem environment from the Sutton and Barto
  Reinforcement Learning: An Introduction Chapter 4.
  For use in the Reinforcement Learning course, Fall 2017, University of Alberta
"""

from utils import rand_norm, rand_in_range, rand_un
import numpy as np

num_total_states = 1000 # num_total_states: integer
current_state = None
seed = None

def env_init():
    global current_state
    current_state = 500


def env_start():
    """ returns numpy array """
    global current_state

    current_state = 500

    return current_state

def env_step(action):
    """
    Arguments
    ---------
    action : int
        the action taken by the agent in the current state

    Returns
    -------
    result : dict
        dictionary with keys {reward, state, isTerminal} containing the results
        of the action taken
    """
    global current_state
    change = 0
    while change == 0:
        change = np.random.RandomState.randint(seed,-100,high = 101)
    current_state += change
    reward = 0
    is_terminal = False
    if current_state <= 0:
        reward = -1
        is_terminal = True
        current_state = 0
    if current_state > 1000:
        reward = 1
        is_terminal = True
        current_state = 1000
    result = {"reward": reward, "state": current_state, "isTerminal": is_terminal}
    return result

def env_cleanup():
    #
    return

def env_message(in_message): # returns string, in_message: string
    """
    Arguments
    ---------
    inMessage : string
        the message being passed

    Returns
    -------
    string : the response to the message
    """
    global seed
    if in_message == "change seed":
        seed = input("Seed of experiment: ")
        seed = np.random.RandomState(seed)
    return ""
