#!/usr/bin/env python

"""
  Author: Adam White, Matthew Schlegel, Mohammad M. Ajallooeian, Sina Ghiassian
  Purpose: Skeleton code for Monte Carlo Exploring Starts Control Agent
           for use on A3 of Reinforcement learning course University of Alberta Fall 2017

"""

from utils import rand_in_range, rand_un
import numpy as np
import pickle
from tiles3 import tiles, IHT

num_states = 1001

max_size = 65536
tile_length = 0.2
iht = IHT(max_size)
w = [0]*max_size
num_tilings = 50
alpha = 0.1/50

last_state = None
discount = 1

def mytiles(x):
    tile_length = 0.2
    s = float(x/1000)
    return tiles(iht, num_tilings,[x*(1/tile_length)])

def agent_init():
    """
    Hint: Initialize the variables that need to be reset before each run begins
    Returns: nothing
    """
    global w
    w = [0 for t in range(max_size)]
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
    tiles = mytiles(last_state)
    v_s = v_hat(state,w)
    v_ls = v_hat(last_state,w)
    for t in tiles:
        w[t] = w[t] + alpha*(reward + discount*(v_s) - v_ls)
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
    tiles = mytiles(last_state)
    v_ls = v_hat(last_state,w)
    for t in tiles:
        w[t] = w[t] + alpha*(reward - v_ls)
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
            value = v_hat(i,w)
            v.append(value)
        return v
    else:
        return "I don't know what to return!!"

def v_hat(state, w):
    tiles = mytiles(state)
    value = 0
    for t in tiles:
        value += w[t]
    return value
