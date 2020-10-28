#!/usr/bin/env python

"""
  Author: Adam White, Matthew Schlegel, Mohammad M. Ajallooeian, Andrew
  Jacobsen, Victor Silva, Sina Ghiassian
  Purpose: Implementation of the interaction between the Gambler's problem environment
  and the Monte Carlon agent using RL_glue.
  For use in the Reinforcement Learning course, Fall 2017, University of Alberta

"""

from rl_glue import *  # Required for RL-Glue
RLGlue("gambler_env", "state_aggregation")

from rndmwalk_policy_evaluation import compute_value_function
import numpy as np
import pickle

if __name__ == "__main__":
    num_episodes = 2000
    max_steps = 10000

    num_runs = 10
    RL_env_message("change seed")
    n = [[] for e in range(num_episodes)]
    error = [[] for e in range(num_episodes)]
    value = []

    for run in range(num_runs):
      counter = 0
      print "run number: ", run
      RL_init()
      print "\n"
      for episode in range(num_episodes):
        RL_episode(max_steps)
        counter += 1
        n[episode].append(RL_agent_message("ValueFunction"))
      RL_cleanup()

    print("Calculating Value Function")
    value = np.load("TrueValueFunction.npy")
    print("Calculating Error")
    for episode in range(len(n)):
        for r in range(num_runs):
            error[episode].append(np.sqrt(1.0/1000.0*np.sum(np.power(np.subtract(value,n[episode][r]),2))))
        error[episode] = np.average(error[episode])

    np.save("ValueFunction2",error)
