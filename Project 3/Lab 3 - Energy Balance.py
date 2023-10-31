#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is the code for Lab 3 of CLIMATE 410 where we are looking at the energy 
balance of earth to come up with a set of equations for a n-number of layers. 
Then using this basic model, apply it to the scenario of another planet (Venus)
and nuclear winter on earth .
"""
# i = row
# j = column
#layer 0 is the surface/ground

import os
import numpy as np
import matplotlib.pyplot as plt


def energybalance(nlayers, emiss=1, emiss_g=1, S0 = 1350, albedo = 0.33):
    '''
    This function calculates the surface and layer temperature of a planet 
    energy balance given the number of layers in the atmosphere and optionally 
    a few other keyword arguments
    
    Parameters
    -----------
    nlayers - int that will determine how many layers that atmosphere will have
    for the calculations. It will also set up the bounds for the matrix that
    all the values are stored and calulated.
    
    emiss - float from (0-1], set at 1. This varies how much energy will be 
    absorbed and then emitted back out of a layer in the atmosphere
    
    emiss_g - float from 0-1 set at 1. This works same as emiss but defined
    just for the surface because it typically only ever is a value of 1
    
    S0 - float that determines how many Wm^-2 of flux that will be coming to a
    planet from the sun. The value that is preset (1350) is recommended in the 
    lab background and scientific questions
    
    albedo - float from [0-1), set at 0.33, an excepted average value for 
    earth. Determines how much of the incoming solar radiation goes back out to
    space from reflection
    
    Returns
    ---------
    layer_temperatures - array of floats that are the temperatures for each 
    layer in the atmosphere and the surface temperature
    
    '''
    
    sigma = 5.67*10**-8
    
    A = np.zeros([nlayers+1, nlayers+1])
    b = np.zeros(nlayers+1)
    
    for i in range(nlayers+1):
        for j in range(nlayers+1):
            if i==j and i==0:
                A[i,j]= -1
                
            elif i==j:
                A[i,j] = -2
           
            elif i==0:
                A[i,j] = (1-emiss)**(np.abs(j-i)-1)
            
            else:
                A[i,j] = emiss*(1-emiss)**(np.abs(j-i)-1)
    
    b[0] = (-S0/4)*(1-albedo)
    Ainv = np.linalg.inv(A)
    fluxes = np.matmul(Ainv,b)
    #print (fluxes)
    surface_temp = (fluxes[0]/(sigma*emiss_g))**(1/4)
    layer_temperatures = (fluxes/(sigma*emiss))**(1/4) 
    layer_temperatures[0] = surface_temp
    
    return layer_temperatures

##########################Question 3 Testing###################################

###Varying the emissivity
emiss_range = np.arange(0.01,1.01,0.01)
surface_temps=np.zeros(len(emiss_range))
for i in range(len(emiss_range)):
    temps = energybalance(1, emiss = emiss_range[i])
    surface_temps[i] = temps[0]
    
#figure out where the matching emissivity is
closest_temp = surface_temps[min(range(len(surface_temps)), key = lambda i: abs(surface_temps[i]-288))]
emissivity = np.where(surface_temps==closest_temp)



#Check for a folder to hold the replicated figures, and if not make one
if not os.path.exists("./Replicated Figures/"):
    os.mkdir("./Replicated Figures/")
    
###Plotting for emissivity changes
fig, ax = plt.subplots(1, 1, figsize=(10, 8))
ax.plot(surface_temps,emiss_range)

#formatting
ax.set_title('Altering Emissivity - Temperatures VS Emissivity Values', 
              fontsize=17)
ax.set_xlabel('Temperature (K)', fontsize=15)
ax.set_ylabel('Emissivity', fontsize=15)
ax.annotate('Emissivity Value to get to 288K on the surface = {}'.format(round(emiss_range[emissivity[0]][0],3)), 
            xy=(0.22,-0.13), xycoords='axes fraction', fontsize=12)
fig.savefig('./Replicated Figures/Varying Emissivity.png')



###Varying layers
layers_range = np.arange(1,11,1)
surface_temps = np.zeros(len(layers_range))
for i in range(len(layers_range)):
    temps = energybalance(layers_range[i], emiss = .255)
    surface_temps[i] = temps[0]

#figure out where the matching layer amount is
closest_temp = surface_temps[min(range(len(surface_temps)), key = lambda i: abs(surface_temps[i]-288))]
layers = np.where(surface_temps==closest_temp)
#print(surface_temps)




###Plotting for changing layers
fig2, ax = plt.subplots(1, 1, figsize=(10, 8))
ax.plot(surface_temps,layers_range)

#formating
ax.set_title('Altering Layers - Surface Temperatures VS Number of Total Layers', 
              fontsize=17)
ax.set_xlabel('Temperature (K)', fontsize=15)
ax.set_ylabel('Layers', fontsize=15)
ax.annotate('Number of Layers to get to 288K on the surface = {}'.format(round(layers_range[layers[0]][0],2)), 
            xy=(0.22,-0.13), xycoords='axes fraction', fontsize=12)
fig2.savefig('./Replicated Figures/Varying Layers Earth.png')



###Atmospheric profile with 5 layers (because that is what the above output gave)
layer = 5
five_layertemps = energybalance(layer, emiss =.255)

fig3, ax = plt.subplots(1, 1, figsize=(10, 8))
ax.plot(five_layertemps,np.arange(0,layer+1))

#formatting
ax.set_title('5-Layer Atmosphere for a Surface Temp of 288K', 
              fontsize=17)
ax.set_xlabel('Temperature (K)', fontsize=15)
ax.set_ylabel('Layer', fontsize=15)

fig3.savefig('./Replicated Figures/5-layer Atmosphere Earth - Normal.png')

##########################Question 4 Venus#####################################

###Varying layers
layers_range = np.arange(1,50,1)
surface_temps = np.zeros(len(layers_range))
for i in range(len(layers_range)):
    #using earths albedo
    temps = energybalance(layers_range[i], S0=2600)
    surface_temps[i] = temps[0]

#figure out where the matching layer amount is
closest_temp = surface_temps[min(range(len(surface_temps)), key = lambda i: abs(surface_temps[i]-700))]
layer = np.where(surface_temps==closest_temp)
#print(layer[0,0])



### Plotting for it
fig4, ax = plt.subplots(1, 1, figsize=(10, 8))
ax.plot(surface_temps,layers_range)
    
#formating
ax.set_title('Venus - Surface Temperatures VS Number of Total Layers', 
              fontsize=17)
ax.set_xlabel('Temperature (K)', fontsize=15)
ax.set_ylabel('Layers', fontsize=15)
ax.annotate('Number of layers to get to 700K on the surface = {}'.format(layer[0][0]), 
            xy=(0.22,-0.13), xycoords='axes fraction', fontsize=12)

fig4.savefig('./Replicated Figures/Varying Layers Venus.png')

##########################Question 5 Testing###################################

def nuclear_winter(nlayers, emiss=1, emiss_g=1, S0 = 1350, albedo = 0.33):
    '''
    This function calculates the surface and layer temperature of a planet 
    energy balance given the number of layers in the atmosphere and optionally 
    a few other keyword arguments. *** What is different in this model is where
    the solar energy in the shortwave spectrum gets absorbed. In this scenario,
    it is the top most layer instead of the surface. The only thing that 
    changes is the b array for incoming solar radiation. The parmeters and 
    returns remain the same and are copied from the energy_balance function 
    for clarifcation. 
    
    Parameters
    -----------
    nlayers - int that will determine how many layers that atmosphere will have
    for the calculations. It will also set up the bounds for the matrix that
    all the values are stored and calulated.
    
    emiss - float from (0-1], set at 1. This varies how much energy will be 
    absorbed and then emitted back out of a layer in the atmosphere
    
    emiss_g - float from 0-1 set at 1. This works same as emiss but defined
    just for the surface because it typically only ever is a value of 1
    
    S0 - float that determines how many Wm^-2 of flux that will be coming to a
    planet from the sun. The value that is preset (1350) is recommended in the 
    lab background and scientific questions
    
    albedo - float from [0-1), set at 0.33, an excepted average value for 
    earth. Determines how much of the incoming solar radiation goes back out to
    space from reflection
    
    Returns
    ---------
    layer_temperatures - array of floats that are the temperatures for each 
    layer in the atmosphere and the surface temperature
    
    '''
    
    sigma = 5.67*10**-8
    
    A = np.zeros([nlayers+1, nlayers+1])
    b = np.zeros(nlayers+1)
    
    for i in range(nlayers+1):
        for j in range(nlayers+1):
            if i==j and i==0:
                A[i,j]= -1
                
            elif i==j:
                A[i,j] = -2
           
            elif i==0:
                A[i,j] = (1-emiss)**(np.abs(j-i)-1)
            
            else:
                A[i,j] = emiss*(1-emiss)**(np.abs(j-i)-1)
    
    ##the big change###
    b[-1] = (-S0/4)*(1-albedo)
    
    Ainv = np.linalg.inv(A)
    #print(A)
    #print(Ainv)
    print(b)
    fluxes = np.matmul(Ainv,b)
    #print (fluxes)
    surface_temp = (fluxes[0]/(sigma*emiss_g))**(1/4)
    layer_temperatures = (fluxes/(sigma*emiss))**(1/4) 
    layer_temperatures[0] = surface_temp
    
    return layer_temperatures

layers = 5

#because the top layer is all absorbing of solar radiation, albedo must go to 0
temps = nuclear_winter(layers, emiss = 0.5, albedo=0)

### Plotting for it

fig5, ax2 = plt.subplots(1, 1, figsize=(10, 8))
ax2.plot(temps,np.arange(0,layers+1))
#formating
ax2.set_title('Nuclear Winter Scenario - Temperatures VS Layers', 
              fontsize=17)
ax2.set_xlabel('Temperature (K)', fontsize=15)
ax2.set_ylabel('Layer', fontsize=15)

fig5.savefig('./Replicated Figures/Nuclear Winter Atmosphere & Surface.png')

 
    