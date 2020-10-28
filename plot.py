#!/usr/bin/env python

"""
 Author: Adam White, Matthew Schlegel, Mohammad M. Ajallooeian, Sina Ghiassian, Zach Holland
 Purpose: for use of Rienforcement learning course University of Alberta Fall 2017
"""

import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
   V = np.load('ValueFunction.npy')
   N = np.load('ValueFunction1.npy')
   Q = np.load('ValueFunction2.npy')
   plt.show()
   plt.plot(V,label = "tile coding")
   plt.plot(N,label = "tabular")
   plt.plot(Q,label = "state aggregation")
   plt.xlabel('Episodes')
   plt.ylabel('MSVE')
   plt.legend()
   plt.show()
