#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is the code for Lab 2 of CLIMATE 410 looking at how to make a 1st order 
ODE solver (Euler Method) and applying it to the competition and predator-prey
equations/relationships
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

#Functions for Lotka-Voterra Competition and Predator Prey Equations

def derivative_competiton(t, N, a=1, b=2, c=1, d=3):
    '''
    The function creates the Lotka-Voterra competition equations for two 
    species with four coefficients for population growth and decline. Returns 
    two derivatives, one for each species
    
    
    Parameters
    -----------
    t - float that will be needed for the lovely scipy situation
    
    N - two element list [N1,N2] the population amounts
    
    a,b,c,d - float, coeffients for the population growth and decline
              values you set to ...
              *** These values are recommended by the lab 2 background/
              description
    
    Returns
    ---------
    dn1dt & dn2dt - floats
        The derviatives that the function creates for the two species
    
    '''
    dn1dt = a*N[0] * (1 - N[0]) - b*N[0] * N[1]
    dn2dt = c*N[1] * (1 - N[1]) - d*N[1] * N[0]
    
    return dn1dt, dn2dt


def derivative_predator(t, N, a=1, b=2, c=1, d=3):
    '''
    The function creates the Lotka-Voterra predator-prey equations for two 
    species with four coefficients for population growth and decline. Returns 
    two derivatives, one for each species
    
    
    Parameters
    -----------
    t - float that will be needed for the lovely scipy RK8 situation
    
    N - two element list [N1,N2] the population amounts
    
    a,b,c,d - float, coeffients for the population growth and decline
              values you set to a=1, b=2, c=1, d=3
              *** These values are recommended by the lab 2 background/
              description
    
    Returns
    ---------
    dn1dt & dn2dt - floats
        The derviatives that the function creates for the two species
    
    '''
    dn1dt = a*N[0] - b*N[0]*N[1]
    dn2dt = -c*N[1] + d*N[0]*N[1]
    
    return dn1dt, dn2dt


# Function for the Euler Solver

def euler_solve(function, n1_init = 0.3, n2_init = 0.6, tstep=1, tstop=100):
     '''
     The function computes the euler method for a function with two varaibles.
     In this case, it will be used for the is for the Lotka-Voterra competition 
     and predator_prey equations
     
     
     Parameters
     -----------
     function - a derivative function that will give you two solution outputs.
     In this case, the options are the derivative_competition or 
     derivative_predator
     
     n1_init - float between 0-1 that determines the starting amount of the 
     population
     
     n2_init - float between 0-1 that determines the starting amount of the 
     second/competing population
     
     tstep - float, determines how small the steps between each calculation are
     
     tstop - float, how many (in this case) years the simulation will stop at
     
     
     Returns
     ---------
     time - array of time for both the equations (since they have to match up)
     
     n1sol - solution for the first population
     
     n2sol - solution for the first population
     
     tstep - float, pulling through this value to properly label the graphs
     
     '''
     
     # Create time array. We won't use that here, but will return it
     # to the caller for convenience.
     t = np.arange(0, tstop, tstep)

     # Create container for both the species solutions and set the initial 
     #conditions for both.
     n1_fun = np.zeros(t.size)
     n1_fun[0] = n1_init
     
     n2_fun = np.zeros(t.size)
     n2_fun[0] = n2_init
     

     # Integrate forward for n1:
     for i in range(1, t.size):
         dN1, dN2 = function(i,[n1_fun[i-1],n2_fun[i-1]])
         n1_fun[i] = n1_fun[i-1] + tstep * dN1
         n2_fun[i] = n2_fun[i-1] + tstep * dN2

     return t, n1_fun, n2_fun, tstep
     

# Function for the rk8 solver
def rk8_solve(function, n1_init = 0.3, n2_init = 0.6, tstep= 1, tstop= 100, 
              a=1, b=2, c=1, d=3):
    '''
    The function computes the rk8 method for a function with two varaibles.
    In this case, it will be used for the is for the Lotka-Voterra competition 
    and predator_prey equations. It also uses the solve_ivp from the scipy 
    toolkit
    
    
    Parameters
    -----------
    function - a derivative function that will give you two solution outputs.
              In this case, the options are the derivative_competition or 
              derivative_predator
    
    n1_init - float between 0-1 that determines the starting amount of the 
              population
    
    n2_init - float between 0-1 that determines the starting amount of the 
              second/competing population
    
    tstep - float, determines how small the steps between each calculation are
    
    tstop - float, how many (in this case) years the simulation will stop at
    
    a,b,c,d - float, coeffients for the population growth and decline
              values you set to a=1, b=2, c=1, d=3
              *** These values are recommended by the lab 2 background/
              description
    
    
    Returns
    ---------
    time - array of time for both the equations (since they have to match up)
    
    n1sol - solution for the first population
    
    n2sol - solution for the first population
    
    max_step - float, pulling through this value to properly label the graphs
    
    '''

    
    result = solve_ivp(function, [0, tstop], [n1_init, n2_init],
                       args=[a,b,c,d], method='DOP853',max_step=tstep)
        
    # Perfom the integration
    t, n1, n2, max_step = result.t, result.y[0,:], result.y[1,:], tstep
        
    #Return values
    return t, n1, n2, max_step


###############################################################################    

#Check for a folder to hold the replicated figures, and if not make one
if not os.path.exists("./Replicated Figures/"):
    os.mkdir("./Replicated Figures/")

# Function competition model
t, n1compeu, n2compeu, tstep = euler_solve(derivative_competiton)
time, n1comprk, n2comprk, max_step = rk8_solve(derivative_competiton)
    
# Plotting 
fig, ax = plt.subplots(1, 1, figsize=(10, 8))
ax.plot(t,n1compeu, color = 'steelblue', linestyle ='solid', 
        label = 'N1 Euler')
ax.plot(t,n2compeu, color = 'red', linestyle ='solid',  label = 'N2 Euler')
ax.plot(time,n1comprk, color = 'steelblue', linestyle ='dotted', 
        label = 'N1 RK8')
ax.plot(time,n2comprk, color = 'red', linestyle ='dotted', label = 'N2 RK8')

# Proper labeling
ax.set_title('Lotka-Volterra Competition Model')
ax.set_xlabel('Time (Years)')
ax.set_ylabel('Population/Carrying Capacity')
ax.legend()
# The labeling for the coefficients and timestep on the graph
ax.tick_params(labelsize = 14)
ax.annotate('Coefficients: a=1, b=2, c=1, d=3', xy=(0.35,-0.13), 
            xycoords='axes fraction', fontsize=12)
ax.annotate('Timestep for Euler: {} (yr)'.format(tstep), xy=(0,-0.13), 
            xycoords='axes fraction', fontsize=12)
ax.annotate('Timestep for RK8: {} (yr)'.format(max_step), xy=(0.70,-0.13), 
            xycoords='axes fraction', fontsize=12)
fig.tight_layout()
fig.savefig('./Replicated Figures/Competition.png')


#Function predator-prey model
t, n1predeu, n2predeu, tstep2= euler_solve(derivative_predator,tstep =0.05)
time, n1predrk, n2predrk, max_step2 = rk8_solve(derivative_predator)

fig2,ax2 = plt.subplots(1,1, figsize=(10,8))
ax2.plot(t,n1predeu, color = 'steelblue', linestyle ='solid', 
         label = 'N1 Euler')
ax2.plot(t,n2predeu, color = 'red', linestyle ='solid', label = 'N2 Euler')
ax2.plot(time,n1predrk, color = 'steelblue', linestyle ='dotted', 
         label = 'N1 RK8')
ax2.plot(time,n2predrk, color = 'red', linestyle ='dotted', label = 'N2 RK8')

#Proper labeling
ax2.set_title('Lotka-Volterra Predator-Prey Model')
ax2.set_xlabel('Time (Years)')
ax2.set_ylabel('Population/Carrying Capacity')
ax2.legend()
# The labeling for the coefficients and timestep on the graph
ax2.annotate('Coefficients: a=1, b=2, c=1, d=3', xy=(0.35,-0.13), 
            xycoords='axes fraction', fontsize=12)
ax2.annotate('Timestep for Euler: {} (yr)'.format(tstep2), xy=(0,-0.13), 
            xycoords='axes fraction', fontsize=12)
ax2.annotate('Timestep for RK8: {} (yr)'.format(max_step2), xy=(0.70,-0.13), 
            xycoords='axes fraction', fontsize=12)
fig2.tight_layout()
fig2.savefig('./Replicated Figures/Predator-Prey.png')



###Varying the time step to show differences in the Euler and RK8##############

def timestep_change():
    '''
    This function is to just save time when running this script because its
    easier to comment out the one line when calling it to run then all these 
    lines in here.
    
    But what this whole process does is take three different time steps of the
    competition and pthe redator-prey function and put them on a plot together 
    so its easy to compare
  
    '''
    #Competition Graph
    smt, smn1compeu, smn2compeu, smtstep = euler_solve(derivative_competiton, 
                                                       tstep = 0.01)
    smtime, smn1comprk, smn2comprk, smmax_step = rk8_solve(derivative_competiton, 
                                                           tstep = 0.01)
    
    medt, medn1compeu, medn2compeu, medtstep = euler_solve(derivative_competiton, 
                                                           tstep = 0.1)
    medtime, medn1comprk, medn2comprk, medmax_step = rk8_solve(derivative_competiton, 
                                                               tstep = 0.1)
    
    lrt, lrn1compeu, lrn2compeu, lrtstep = euler_solve(derivative_competiton,
                                                       tstep = 1)
    lrtime, lrn1comprk, lrn2comprk, lrmax_step = rk8_solve(derivative_competiton, 
                                                           tstep = 1)
    
    fig5,ax5 = plt.subplots(1,1, figsize=(10,8))
    ax5.plot(smt,smn1compeu, color = 'blue', linestyle ='solid', 
             label = 'Small Euler tstep {}'.format(smtstep))
    ax5.plot(smtime,smn1comprk, color = 'blue', linestyle ='dotted', 
             label = 'Small RK8 max tstep {}'.format(smmax_step))
    ax5.plot(medt,medn1compeu, color = 'red', linestyle ='solid', 
             label = 'Medium Euler tstep {}'.format(medtstep))
    ax5.plot(medtime,medn1comprk, color = 'red', linestyle ='dotted', 
             label = 'Medium RK8 max tstep {}'.format(medmax_step))
    ax5.plot(lrt,lrn1compeu, color = 'green', linestyle ='solid', 
             label = 'Large Euler tstep {}'.format(lrtstep))
    ax5.plot(lrtime,lrn1comprk, color = 'green', linestyle ='dotted', 
             label = 'Large RK8 max tstep {}'.format(lrmax_step))
    
    #Proper labeling
    ax5.set_title('Lotka-Volterra Competition Model Time Step Change')
    ax5.set_xlabel('Time (Years)')
    ax5.set_ylabel('Population/Carrying Capacity')
    ax5.legend()
    # The labeling for the coefficients and timestep on the graph
    ax5.annotate('Coefficients: a=1, b=2, c=1, d=3', xy=(0.35,-0.13), 
                xycoords='axes fraction', fontsize=12)
    fig5.tight_layout()
    
    if not os.path.exists("./Competition Figures/"):
        os.mkdir("./Competition Figures/")
        
    fig5.savefig('./Competition Figures/Competition Time Step Change.png')
    
    
    
    #Predator-Prey relationship
    
    smt, smn1predeu, smn2predeu, smtstep = euler_solve(derivative_predator,
                                                       tstep = 0.0005)
    smtime, smn1predrk, smn2predrk, smmax_step = rk8_solve(derivative_predator, 
                                                           tstep = 0.0005)
    
    medt, medn1predeu, medn2predeu, medtstep = euler_solve(derivative_predator, 
                                                           tstep = 0.005)
    medtime, medn1predrk, medn2predrk, medmax_step = rk8_solve(derivative_predator, 
                                                               tstep = 0.005)
    
    lrt, lrn1predeu, lrn2predeu, lrtstep = euler_solve(derivative_predator, 
                                                       tstep = 0.05)
    lrtime, lrn1predrk, lrn2predrk, lrmax_step = rk8_solve(derivative_predator,
                                                           tstep = 0.05)
    
    fig6,ax6 = plt.subplots(1,1, figsize=(10,8))
    ax6.plot(smt,smn1predeu, color = 'blue', linestyle ='solid', 
             label = 'Small Euler tstep {}'.format(smtstep))
    ax6.plot(smtime,smn1predrk, color = 'blue', linestyle ='dotted', 
             label = 'Small RK8 max tstep {}'.format(smmax_step))
    ax6.plot(medt,medn1predeu, color = 'red', linestyle ='solid', 
             label = 'Medium Euler tstep {}'.format(medtstep))
    ax6.plot(medtime,medn1predrk, color = 'red', linestyle ='dotted', 
             label = 'Medium RK8 max tstep {}'.format(medmax_step))
    ax6.plot(lrt,lrn1predeu, color = 'green', linestyle ='solid', 
             label = 'Large Euler tstep {}'.format(lrtstep))
    ax6.plot(lrtime,lrn1predrk, color = 'green', linestyle ='dotted', 
             label = 'Large RK8 max tstep {}'.format(lrmax_step))
    
    #Proper labeling
    ax6.set_title('Lotka-Volterra Predator-Prey Model Time Step Change')
    ax6.set_xlabel('Time (Years)')
    ax6.set_ylabel('Population/Carrying Capacity')
    ax6.legend()
    # The labeling for the coefficients and timestep on the graph
    ax6.annotate('Coefficients: a=1, b=2, c=1, d=3', xy=(0.35,-0.13), 
                xycoords='axes fraction', fontsize=12)
    fig6.tight_layout()
    
    if not os.path.exists("./Predator-Prey Figures/"):
        os.mkdir("./Predator-Prey Figures/")
    fig6.savefig('./Predator-Prey Figures/Predator-Prey Time Step Change.png')

####### Un-comment if you want to run this :)
#timestep_change()

##############################Question 2#######################################

#Using only RK8 because it is the higher order and more accurate form

#Putting the coefficents in an array to use for plotting label purposes 
coeff = [0.05,1,0.05,2]

time, n1comprk, n2comprk, max_step = rk8_solve(derivative_competiton, 
    n1_init = 0.3, n2_init = 0.6, tstep= 1, tstop= 100, 
    a=coeff[0], b=coeff[1], c=coeff[2], d=coeff[3])

#Plotting
fig3,ax3 = plt.subplots(1,1, figsize=(10,8))
ax3.plot(time,n1comprk, color = 'steelblue', linestyle ='dotted', 
         label = 'N1 RK8')
ax3.plot(time,n2comprk, color = 'red', linestyle ='dotted', label = 'N2 RK8')

#Proper labeling
ax3.set_title('Lotka-Volterra Competition Model - Stable Equilibrium')
ax3.set_xlabel('Time (Years)')
ax3.set_ylabel('Population/Carrying Capacity')
ax3.legend()
# The labeling for the coefficients and timestep on the graph
ax3.annotate('Coefficients: a={}, b={}, c={}, d={}'.format(coeff[0],coeff[1],
            coeff[2],coeff[3]), xy = (0.35,-0.13), xycoords='axes fraction', 
            fontsize=12)
ax3.annotate('Timestep for RK8: {} (yr)'.format(max_step2), xy=(0.70,-0.13), 
            xycoords='axes fraction', fontsize=12)
fig3.tight_layout()


fig3.savefig('./Competition Figures/Stable Equilibrium Competition Coeff \
             {},{},{},{}.png'.format(coeff[0],coeff[1],coeff[2],coeff[3]))

##############################Question 3#######################################

#Continuing to only use RK8 because it is higher order and better

coeff = [1,1,1,3]
time, n1predrk, n2predrk, max_step = rk8_solve(derivative_predator, 
    n1_init = 0.5, n2_init = 0.4, tstep= 0.5, tstop= 100, 
    a=coeff[0], b=coeff[1], c=coeff[2], d=coeff[3])

#Plotting
fig3,ax3 = plt.subplots(1,1, figsize=(10,8))
ax3.plot(time,n1predrk, color = 'steelblue', linestyle ='dotted', 
         label = 'N1 RK8')
ax3.plot(time,n2predrk, color = 'red', linestyle ='dotted', label = 'N2 RK8')
#Proper labeling
ax3.set_title('Lotka-Volterra Predator-Prey Model')
ax3.set_xlabel('Time (Years)')
ax3.set_ylabel('Population/Carrying Capacity')
ax3.legend()
# The labeling for the coefficients and timestep on the graph
ax3.annotate('Coefficients: a={}, b={}, c={}, d={}'.format(coeff[0],coeff[1],
            coeff[2],coeff[3]), xy = (0.35,-0.13), xycoords='axes fraction', 
            fontsize=12)
ax3.annotate('Timestep for RK8: {} (yr)'.format(max_step2), xy=(0.70,-0.13), 
            xycoords='axes fraction', fontsize=12)
fig3.tight_layout()

fig3.savefig('./Predator-Prey Figures/Predator-Prey Conditions Change Coeff \
             {},{},{},{}.png'.format(coeff[0],coeff[1],coeff[2],coeff[3]))       

#Phase diagram situation 
fig4,ax4 = plt.subplots(1,1, figsize=(10,8))      
ax4.plot(n1predrk,n2predrk, color = 'steelblue', linestyle ='solid', 
         label = 'Phase Diagram') 
#Proper labeling       
ax4.set_title('Lotka-Volterra Predator-Prey Model - Phase Diagram')
ax4.set_xlabel('Prey Population')
ax4.set_ylabel('Predator Population')
# Already checked for this directory 
fig4.savefig('./Predator-Prey Figures/Predator-Prey Phase Diagram Coeff \
             {},{},{},{}.png'.format(coeff[0],coeff[1],coeff[2],coeff[3]))       
        
        
        
        
        
        
        
        
        
        
        



