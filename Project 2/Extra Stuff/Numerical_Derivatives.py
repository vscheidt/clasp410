#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 09:50:04 2023

@author: torischeidt
"""

import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0,4*np.pi,0.5)
fx = np.sin(x)
dfdx_sol = np.cos(x)


#create result array, fill with zeros,
dfdx= np.zeros(fx.size)
dfdx[:-1] = (fx[1:]-fx[:-1]) /dx
dfdx[-1] = dfdx[-2]

plt.plot(x,dfdx)
plt.plot(x,fx)













