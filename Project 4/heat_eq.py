#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is a script for CLIMATE 410 Lab 4: Heat Diffusion & Permafrost. We solve
for the heat diffusivity equation numerically with a forward-difference approach
amd apply this equation with some boundary conditions to simulate Kangerlussuaq,
Greenland Permafrost values
"""
import numpy as np
import matplotlib.pyplot as plt

#Solution from the Lab 4 documentation to compare the 1D - Diffusivity 
solution = [[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], 
            [0.64, 0.48, 0.4, 0.32, 0.26, 0.21, 0.17, 0.1375, 0.11125, 0.09, 0.0728125],
            [0.96, 0.8, 0.64, 0.52, 0.42, 0.34, 0.275, 0.2225, 0.18, 0.145625, 0.1178125],
            [0.96, 0.8, 0.64, 0.52, 0.42, 0.34, 0.275, 0.2225, 0.18, 0.145625, 0.1178125],
            [0.64, 0.48, 0.4, 0.32, 0.26, 0.21, 0.17, 0.1375, 0.11125, 0.09, 0.0728125],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

#Values given in Lab 4 documentation to create a continuous model for the temperatures
    #of Kangerlussuaq
t_kanger = np.array([-19.7 ,-21.0 ,-17.0 ,-8.4 ,2.3 ,8.4 ,10.7 ,8.5 ,3.1,
                     -6.0, -12.0, -16.9])

#the proper C value for the greenland problem 
c2_green = .25*((1/1000)**2)*24*60*60

def sample_init(x):
    '''
    A fucnction for the intial stating condtion for the diffusivity equation
    Used to validate the solver 
    
    Parameters
    ----------
    x - float or an array of floats to be calculated for later use in the 
    heat_solve equation
    
    Returns
    --------------
    a float value or float array of the calculated expression 4*x - 4*x**2 
    given the x variable
    
    '''
    return 4*x - 4*x**2

def temp_kanger(t, tempshift = 0):
    '''
    A function for the top boundary conditon for Kangerlussuaq, Grrenland to be
    used in the heat_solve equation
    
    Parameters
    -----------
    t - float or array of floats of time steps in days to perform the calculations
    * also needs an array for of temperatures (t_kanger) 
    previously defined outside the function
    
    tempshift - float used for part 3 to simulate a warming planet
    
    Return
    -----------
    a float value or float array of the calculated expression 
    t_amp*np.sin(np.pi/180 * t -np.pi/2) + t_kanger.mean() given the t variable
    
    '''
    t_amp = ((t_kanger+tempshift) - (t_kanger+tempshift).mean()).max()
    
    return t_amp*np.sin(np.pi/180 * t -np.pi/2) + (t_kanger+tempshift).mean()

def heat_solve(x_step=0.2, t_step=0.02, C2=1.0, xmax= 1.0, tmax= 0.2, 
               init= 0, top= 0, bot= 0, temp_shift= 0, neumann=0):
    """
    This is the main function for solving the heat diffusivity. Its main features
    include setting the boundary conditions, initial conditions, and using a
    numerical forward-difference method. 
    
    
    Parameters
    ------------
    x_step - float, the spacial step (in meters for greenland)
    
    t_step - float, the time step in days
    
    C2 - float, rate of diffusion mm^2/s (must be converted for greenland problem)
    
    xmax - float, the max spacial value (in meters for greenland)
    
    tmax - float, the max time value (in days for greenland)
    
    init - float or function, sets the intital starting conditions for the heat equation
    only function that has been created to go in here is sample_init. Initially set to zero
    
    top - float or a function, sets the top boundary condition for the heat equation
    only function that has been created is temp_kanger. Initially set to zero
    
    bot - float or a function, sets the bottom boundary condition for the heat equation
    no function has been made but the script is ready to recieve a function. Initially set to zero
    
    temp_shift - float used in part 3 to cause a warming change simulating global
    climate change values. Initially set to zero
    
    neumann - boolean, where setting it to true (1) changes the end behavior
    Initially set to zero
    
    
    Returns
    --------------
    x - float array of the spacial steps taken to get to the final answer matrix (temps)
    
    t - float array of the time steps taken to get to the final answer matrix (temps)
    
    temp - float matrix (mxn) (spacial and time steps) that has the temperature
    of of each spacial step and time.
    
    
    """
    #Checking for stability
    if t_step > x_step**2/(2*C2):
        print ('You have an invalid stability. The graphs are going to look whack')
        return
    
    
    # set constants: 
    r =C2 *t_step/x_step**2
    
    # create space and time grids
    x=np.arange(0, xmax+x_step, x_step)
    t=np.arange(0, tmax+t_step, t_step)
    
    #save points
    M,N = x.size, t.size
    
    #create temp solution array
    temp = np.zeros([M,N])
    
    #Set boundary conditions
    if callable(top):
        temp[0,:]= top(t)
        #for part 3 to add the surface temperature warming
        if temp_shift != 0:
            temp[0,:] = top(t, tempshift =temp_shift)
    else:
        temp[0,:]= top
    
    if callable(bot):
        temp[-1,:] = bot(t)
    else:
        temp[-1,:] = bot
    
    
    
    #Set intial conditions
    if callable(init):
       temp[:,0] = init(x)
    else:
        temp[:,0] = init



    #Actually the equations
    for j in range(0,N-1):
        for i in range(1,M-1):
            temp[i, j+1] = (1-2*r)*temp[i,j] + r*(temp[i+1,j]+temp[i-1,j])
        if neumann == 1:
            temp[0,j+1] = temp[1,j+1]
            temp[-1,j+1] = temp[-2,j+1]
     
    
    return x, t, temp
    
#######################Initial checking with test case#########################  
x,t,temp = heat_solve(t_step= 0.02, x_step =0.2, init = sample_init)

#Validating the solution by finding the point with the largest error
max_error=(temp-solution).max()

# Create a figure/axes object
fig, axes = plt.subplots(1, 1)
# Create a color map and add a color bar.
map = axes.pcolor(t, x, temp, cmap='hot', )
plt.colorbar(map, ax=axes, label='Temperature ($^\circ$C)')

#Properly labeling the graphs
plt.title('Validating Solver 1D Heat Diffusivity')
plt.xlabel('Time (seconds)')
plt.ylabel('Space Step')

# Putting the largest error on the graph
axes.annotate('Max Error: {}'.format(max_error), xy=(0.2,-0.22), 
            xycoords='axes fraction', fontsize=9)

#Purposely triggering the stability criteria to check that it works
#print('Checking if the stability criteria works. Another statement should pop up')
#x,t,temp = heat_solve(t_step= 0.2, x_step =0.2, init = sample_init)


################################ Greenland ###################################
def plot_temp(x, time, temp, axes, xlabel='Time ($s$)', title='',
              ylabel='Distance ($m$)', clabel=r'Temperature ($^{\circ} C$)',
              cmap='inferno', **kwargs):
    '''
    Add a pcolor plot of the heat equation to `axes`. Add a color bar.

    Parameters
    ----------
    x
        Array of position values
    time
        Array of time values 
    temp
        array of temperature solution
    xlabel, ylabel, title, clabel
        Axes labels and titles
    cmap : inferno
        Matplotlib colormap name.
    '''

    map = axes.pcolor(time, x, temp, cmap=cmap, **kwargs)
    plt.colorbar(map, ax=axes, label=clabel)
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_title(title)
    axes.invert_yaxis()

def plot_line(x, time, temp, axes, xlabel='Time ($s$)', title='',
              ylabel='Distance ($m$)', inverty=False,**kwargs):
    '''
    Add a pcolor plot of the heat equation to `axes`. Add a color bar.

    Parameters
    ----------
    x
        Array of position values
    time
        Array of time values 
    temp
        array of temperature solution
    xlabel, ylabel, title, clabel
        Axes labels and titles
    cmap : inferno
        Matplotlib colormap name.
    '''
    
    loc = int(-365/10) # Final 365 days of the result.
    # Extract the min values over the final year:
    winter = temp[:, loc:].min(axis=1)
    summer = temp[:, loc:].max(axis=1)
    
    
    plt.plot(winter, x, label='Winter')
    plt.plot(summer, x, '--', label='Summer')
    #plt.plot(winter, x2, label='Winter 2')
    #plt.plot(summer, x2, '--', label='Summer 2')
    plt.xlim(-8,4)
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_title(title)
    axes.invert_yaxis()
    plt.legend()
    
    if inverty==True:
        axes.invert_yaxis()
    
    #Finding where the lines intersect and where the summer is zero
    for i in range(0,len(summer)):
        if summer[i] <= 0:
            index_zero = i
            break
    ax2.annotate('Active Layer Depth: {} m'.format(x_green[index_zero]), xy=(0.15,-0.14), 
                xycoords='axes fraction', fontsize=12)

    for j in range(0,len(summer)):
        if summer[j] - winter[j] <= np.abs(0.1):
            index_perma = j
            break
    for k in range(0,len(winter)):
        if winter[k]>=0:
            winter_zero = k-1
            break
    ax2.annotate('Isothermal Permafrost Range: {} m to {} m'.format(x_green[index_perma],x_green[winter_zero]), 
                 xy=(0.45,-0.14), xycoords='axes fraction', fontsize=12)
    
    
x_green, t_green, temp_green = heat_solve(t_step = 10, x_step = 1,
        C2 = c2_green, xmax = 100,tmax = 45*365, top = temp_kanger, bot = 5)

loc = int(-365/10) # Final 365 days of the result.
# Extract the min values over the final year:
winter = temp_green[:, loc:].min(axis=1)
summer = temp_green[:, loc:].max(axis=1)



fig, axes = plt.subplots(1, 1)

plot_temp(x_green, t_green/365., temp_green, axes, xlabel='Time (Years)', ylabel='Depth ($m$)',
          cmap='seismic', vmin=-25, vmax=25,
          clabel='Temperature ($^\circ$C)', 
          title='Greenland Permafrost Diffusivity')
fig.tight_layout()

fig, ax2 = plt.subplots(1, 1, figsize=(10, 8))

plot_line(x_green, t_green, temp_green, ax2, xlabel='Time ($s$)', title='Kangerlussuaq, Greenland Temperature',
            ylabel='Depth ($m$)')
fig.tight_layout()

#Finding where the lines intersect and where the summer is zero
for i in range(0,len(summer)):
    if summer[i] <= 0:
        index_zero = i
        break
ax2.annotate('Active Layer Depth: {} m'.format(x_green[index_zero]), xy=(0.15,-0.14), 
            xycoords='axes fraction', fontsize=12)

for j in range(0,len(summer)):
    if summer[j] - winter[j] <= np.abs(0.1):
        index_perma = j
        break
for k in range(0,len(winter)):
    if winter[k]>=0:
        winter_zero = k-1
        break

########################### Global Warming Conditions #########################
#1
#-------------------------------------------
warming = np.array([0.5,1,3])

x_green05, t_green, temp_green = heat_solve(t_step = 10, x_step = 1,
        C2 = c2_green, xmax = 100,tmax = 45*365, top = temp_kanger, 
        bot = 5, temp_shift = warming[0])

fig, axes = plt.subplots(1, 1)

plot_temp(x_green05, t_green/365., temp_green, axes, xlabel='Time (Years)', ylabel='Depth ($m$)',
          cmap='seismic', vmin=-25, vmax=25,
          clabel='Temperature ($^\circ$C)', 
          title='Greenland Permafrost Diffusivity +{} $^\circ$C'.format(warming[0]))
fig.tight_layout()

fig, ax2 = plt.subplots(1, 1, figsize=(10, 8))

plot_line(x_green05, t_green, temp_green, ax2, xlabel='Time ($s$)', 
          title='Kangerlussuaq, Greenland Temperature  +{} $^\circ$C'.format(warming[0]),
            ylabel='Depth ($m$)')
fig.tight_layout()

#2
#-------------------------------------------

x_green1, t_green, temp_green = heat_solve(t_step = 10, x_step = 1,
        C2 = c2_green, xmax = 100,tmax = 45*365, top = temp_kanger, 
        bot = 5, temp_shift = warming[1])

fig, axes = plt.subplots(1, 1)

plot_temp(x_green1, t_green/365., temp_green, axes, xlabel='Time (Years)', ylabel='Depth ($m$)',
          cmap='seismic', vmin=-25, vmax=25,
          clabel='Temperature ($^\circ$C)', 
          title='Greenland Permafrost Diffusivity +{} $^\circ$C'.format(warming[1]))
fig.tight_layout()

fig, ax2 = plt.subplots(1, 1, figsize=(10, 8))

plot_line(x_green1, t_green, temp_green, ax2, xlabel='Time ($s$)', 
          title='Kangerlussuaq, Greenland Temperature  +{} $^\circ$C'.format(warming[1]),
            ylabel='Depth ($m$)')
fig.tight_layout()

#3
#------------------------------------------
x_green3, t_green, temp_green = heat_solve(t_step = 10, x_step = 1,
        C2 = c2_green, xmax = 100,tmax = 45*365, top = temp_kanger, 
        bot = 5, temp_shift = warming[2])

fig, axes = plt.subplots(1, 1)

plot_temp(x_green3, t_green/365., temp_green, axes, xlabel='Time (Years)', ylabel='Depth ($m$)',
          cmap='seismic', vmin=-25, vmax=25,
          clabel='Temperature ($^\circ$C)', 
          title='Greenland Permafrost Diffusivity +{} $^\circ$C'.format(warming[2]))
fig.tight_layout()

fig, ax2 = plt.subplots(1, 1, figsize=(10, 8))

plot_line(x_green3, t_green, temp_green, ax2, xlabel='Time ($s$)', 
          title='Kangerlussuaq, Greenland Temperature  +{} $^\circ$C'.format(warming[2]),
            ylabel='Depth ($m$)')
fig.tight_layout()



###Creating a depth and temperature graph that compares the original 
#with the 3 degrees of extra warming#######################################
'''
fig, ax2 = plt.subplots(1, 1, figsize=(10, 8))

plot_line(x_green, x_green3, t_green, temp_green, ax2, xlabel='Time ($s$)', 
          title='Kangerlussuaq, Greenland Temperature  +{} $^\circ$C'.format(warming[0]),
            ylabel='Depth ($m$)')
'''
# =============================================================================
# plot_line(x_green3, t_green, temp_green, ax2, xlabel='Time ($s$)', 
#           title='Kangerlussuaq, Greenland Temperature  +{} $^\circ$C'.format(warming[2]),
#             ylabel='Depth ($m$)',inverty=True)
# fig.tight_layout()
# =============================================================================

# Set indexing for the final year of results:
loc = int(-365/10) # Final 365 days of the result.
# Extract the min values over the final year:
winter_3 = temp_green[:, loc:].min(axis=1)
summer_3 = temp_green[:, loc:].max(axis=1)
# Create a temp profile plot:
fig, ax2 = plt.subplots(1, 1, figsize=(10, 8))
plt.plot(winter_3, x_green3, label=('Winter +{} $^\circ$C').format(warming[2]))
plt.plot(summer_3, x_green3, '--', label=('Summer +{} $^\circ$C').format(warming[2]))
plt.plot(winter, x_green, label='Winter')
plt.plot(summer, x_green, '--', label='Summer')

#Properly labeling the graphs
ax2.invert_yaxis()
plt.title('Kangerlussuaq, Greenland Temperature')
plt.xlabel('Temperature ($^\circ$C)')
plt.xlim(-8,4)
plt.ylabel('Depth (m)')
plt.grid()
plt.legend()

for i in range(0,len(summer_3)):
    if summer_3[i] <= 0:
        index_zero_3 = i
        break
for j in range(0,len(summer_3)):
    if summer_3[j] - winter_3[j] <= np.abs(0.1):
        index_perma_3 = j
        break
for k in range(0,len(winter_3)):
    if winter_3[k]>=0:
        winter_zero_3 = k-1
        break

active_change = index_zero_3 - index_zero

permafrost_change = (winter_zero_3-index_zero_3) - (winter_zero - index_zero)


ax2.annotate('Permafrost Start Depth Difference: {}m & Total Depth Difference {}m'.format(active_change,permafrost_change), 
             xy=(0.15,-0.14), xycoords='axes fraction', fontsize=12)
