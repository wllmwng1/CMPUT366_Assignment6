#!/usr/bin/env python

"""
  Author: Adam White, Matthew Schlegel, Mohammad M. Ajallooeian, Sina Ghiassian
  Purpose: Skeleton code for Monte Carlo Exploring Starts Control Agent
           for use on A3 of Reinforcement learning course University of Alberta Fall 2017

"""

from utils import rand_in_range, rand_un
import numpy as np
import pickle

num_states = 1001
w = []
last_state = None
alpha = 0.1
discount = 1

def agent_init():
    """
    Hint: Initialize the variables that need to be reset before each run begins
    Returns: nothing
    """
    global w
    w = [0 for s in range(num_states/100)]
    #initialize the policy array in a smart way

def agent_start(state):
    """
    Hint: Initialize the variavbles that you want to reset before starting a new episode
    Arguments: state: numpy array
    Returns: action: integer
    """
    # pick the first action, don't forget about exploring starts
    global last_state

    action = 0 #does not matter
    last_state = state

    return action


def agent_step(reward, state): # returns NumPy array, reward: floating point, this_observation: NumPy array
    """
    Arguments: reward: floting point, state: integer
    Returns: action: integer
    """
    # select an action, based on Q
    global w
    global last_state
    index = state/100
    last_index = last_state/100
    if last_index == 10:
        last_index = 9
    if index == 10:
        index = 9
    w[last_index] = w[last_index] + alpha*(reward + discount*(v_hat(index,w)) - v_hat(last_index,w))
    action = 0
    last_state = state
    return action

def agent_end(reward):
    """
    Arguments: reward: floating point
    Returns: Nothing
    """
    # do learning and update pi
    global w
    global last_state
    last_index = last_state/100
    if last_index == 10:
        last_index = 9
    w[last_index] = w[last_index] + alpha*(reward - v_hat(last_index,w))
    return

def agent_cleanup():
    """
    This function is not used
    """
    # clean up
    return

def agent_message(in_message): # returns string, in_message: string
    global w
    """
    Arguments: in_message: string
    returns: The value function as a string.
    This function is complete. You do not need to add code here.
    """
    # should not need to modify this function. Modify at your own risk
    if (in_message == 'ValueFunction'):
        v = [0]
        for i in range(1,num_states):
            index = i/100
            if index == 10:
                index = 9
            v.append(w[index])
        return v
    else:
        return "I don't know what to return!!"

def v_hat(state, w):
    return w[state]
