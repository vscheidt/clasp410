#!/usr/bin/env python3

'''
Coffee problem in class
'''

import numpy as np
import matplotlib.pyplot as plt

# create a time array
tfinal, tstep = 600, 1
time = np.arange(0, tfinal, tstep)


#Solving for temperature
def solve_temp(time, k=1/300, T_env=25, T_int=90):
    '''
    This function takes an array of times and returns an array of temperatures
    corresponding to each time.

    Parameters
    ==============
    time : numpy array of times
        array of time inputs for which you want corresponding temps


    Other Parameters
    ==================

    
    Returns
    ========
    temp = the temperature of the coffee given an amount of time and other
    temperatures (environment and intial)




    '''

    temp = T_env + (T_int - T_env) * np.exp(-k * time)
    
    
    return temp





