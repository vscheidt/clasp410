#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 09:38:31 2023

@author: torischeidt
"""
#Importing all necessary packaages
import numpy as np
import matplotlib.pyplot as plt
import snowball_functions as sbf

#a cute little constant I dont want to put inside the functions now
radearth = 6357000


def gen_grid(npoints=18):
    '''
    This function is a way to create the grids for any number of latitudinal bands
    Parameters
    ----------
    npoints : int, optional
        This is the amount of total amount of latitude values that will be used for the equations
        The default is 18.

    Returns
    -------
    dlat : float
        The change in latitude between two estimated latitude bands
    lats : float
        The actual latitudes that will be used for the calculations 
    edge : float
        The borders of the latitudes in the lats function

    '''
    
    dlat = 180/npoints
    lats = np.linspace(dlat/2.,180-dlat/2.,npoints)
    edge = np.linspace(0,180, npoints+1)
    
    return dlat,lats,edge


def snowballEarf(npoints = 18, dt = 1, tstop = 10000, lamb=100, S0 = 1370):
    '''
    This function completes the graphic that is required to be produced at the end of 
    question/step 1. It plots four different lines that show the different 
    steps of progression for the temperature-time equation.
    
    
    Parameters
    ----------
    npoints : int, optional
        Same as the gen_grid, it helps to determine latitudes. The default is 18.
    dt : int, optional
        Change of time timestep. The default is 1 (year).
    tstop : int, optional
        The total amount of timesteps that will be taken. The default is 10000 (years).
    lamb : float, optional
        The diffusivity constant for the heat equation. The default is 100 (m^2/s).
    S0 : float, optional
        This is the solar constant, used for solar insolation factor. The default is 1370 W/m^2.

    Returns
    -------
    None.

    '''
    
    #All the constants given in the lab spec
    albedo = 0.3
    sigma = 5.67*10**-8 
    rho =1020
    emiss = 1
    c = 4.2*10**6
    mxdlyr =50

    #Creating the grid
    dlat, lats, edges = gen_grid(npoints)
    
    #Set delta y
    dy = np.pi * radearth/ npoints

    #Use one of the given functions in snowball_functions to get intial surface temperatures
    T_warm=sbf.temp_warm(lats)
    #Replicating this orginial array to use different lines of the end product graph
    T_warm_init = T_warm
    T_sphere = T_warm
    T_all = T_warm

    #Tri-diagnal Set-up to start solving for temperature
    A = -2*np.identity(npoints)
    for i in range(1,len(A)-1):
            A[i,i+1] = 1
            A[i,i-1] = 1
    
    #Boundary Conditions
    A[0,1] = 2
    A[-1,-2] = 2
    
    #Changing dt into the same units of lambda and the other constants given
    d_sec = dt*60*60*24*365
    
    #Making the L matrix
    a_matrix = (1/(dy**2))* A
    L = np.identity(npoints) - (lamb*d_sec*a_matrix)
    
    #If the distance isnt 1, makes sure there is an even distribution of points
    nsteps = int(tstop/dt)
    
    #Calculating the Basic Diffusion equation
    for i in range(nsteps):
        T_warm= np.matmul(np.linalg.inv(L),T_warm)
        
    #Central Difference Craziness for the spherical correction
    B = np.zeros((npoints,npoints))
    B[np.arange(npoints-1), np.arange(npoints-1)+1] = 1
    B[np.arange(npoints-1)+1, np.arange(npoints-1)] = -1
    B[0,:] = B[-1,:] = 0
    
    
    #Setting up Insolation:
    insol = sbf.insolation(S0, lats)
    
    #Finding the area of the xz plane and the change between each one
    Axz = np.pi*((radearth+50.0)**2 - radearth**2)*np.sin(np.pi/180. *lats)
    dAxz = np.matmul(B,Axz)/(Axz*4*dy**2)
    
    #Code for the spherical and insolation corrections
    for i in range(nsteps):
        #Just the spherical correction, using the matrices created above
        spherecord = lamb*d_sec*np.matmul(B,T_sphere)*dAxz
        T_sphere += spherecord
        T_sphere = np.matmul(np.linalg.inv(L), T_sphere)
        
        #Spherical and insolation correction
        spherecord_all = lamb*d_sec*np.matmul(B,T_all)*dAxz
        T_all += spherecord_all
            #energy balance equation
        radiative = (1-albedo)*insol-emiss*sigma*(T_all+273)**4
        T_all += d_sec * radiative / (rho*c*mxdlyr)
        T_all = np.matmul(np.linalg.inv(L), T_all)
    
    
    #Plotting and making it all look fabulous!                
    fig, axes = plt.subplots(1, 1)
    axes.plot(lats, T_warm, color='orange', label = 'Basic Diffusion')
    axes.plot(lats, T_warm_init, color='teal',label = 'Initial Condition')
    axes.plot(lats, T_sphere, color='green', label = "Diff + Spherical Correction")
    axes.plot(lats, T_all, color='red', label ='Diff + SphCorr + Radiative')
    plt.title('Lab Graphic For Replication')
    plt.xlabel('Latitude')
    plt.ylabel('Temperature $^\circ$C')
    plt.legend()
    
def snowballEarfWarm(npoints = 18, dt = 1, tstop = 10000, lamb=100, emiss = 1):
    '''
    This function answers question 2 of the lab spec. It will used as proof for
    a choice in emissvity and diffusivity value to get the inital condition
    to look like the line with spherical & insolation factors

    Parameters
    ----------
    npoints : int, optional
        Same as the gen_grid, it helps to determine latitudes. The default is 18.
    dt : int, optional
        Change of time timestep. The default is 1 (year).
    tstop : int, optional
        The total amount of timesteps that will be taken. The default is 10000 (years).
    lamb : float, optional
        The diffusivity constant for the heat equation. The default is 100 (m^2/s).
    emiss : float, 0-1, optional
        The emissivity value of radiating surface of the planet. The default is 1.

    Returns
    -------
    None.

    '''
    #All the constants given in the lab spec
    S0 = 1370
    albedo = 0.3
    sigma = 5.67*10**-8 
    rho =1020
    c = 4.2*10**6
    mxdlyr =50
    
    #Creating the grid
    dlat, lats, edges = gen_grid(npoints)
    
    #Set delta y
    dy = np.pi * radearth/ npoints
    
    #Use one of the given functions in snowball_functions to get intial surface temperatures
    T_all=sbf.temp_warm(lats)
    #Only doing one replication of the array to save the inital conditions for plotting
    T_warm_init = T_all


    #Tri-diagnal Set-up
    A = -2*np.identity(npoints)
    for i in range(1,len(A)-1):
            A[i,i+1] = 1
            A[i,i-1] = 1
    
    #Boundary Conditions
    A[0,1] = 2
    A[-1,-2] = 2
    
    #Changing dt to be in the proper units to be used with lambda and the other constants
    d_sec = dt*60*60*24*365
    
    #Making the L matrix
    a_matrix = (1/(dy**2))* A
    L = np.identity(npoints) - (lamb*d_sec*a_matrix)
    
    #Central Difference Craziness
    B = np.zeros((npoints,npoints))
    B[np.arange(npoints-1), np.arange(npoints-1)+1] = 1
    B[np.arange(npoints-1)+1, np.arange(npoints-1)] = -1
    B[0,:] = B[-1,:] = 0
    
    
    #Setting up Insolation:
    insol = sbf.insolation(S0, lats)
    
    #Making the L matrix
    Axz = np.pi*((radearth+50.0)**2 - radearth**2)*np.sin(np.pi/180. *lats)
    dAxz = np.matmul(B,Axz)/(Axz*4*dy**2)
   
    #If the distance isnt 1, makes sure there is an even distribution of points
    nsteps = int(tstop/dt)
    
    #Loop to calulate the spherical and insolation corrections
    for i in range(nsteps):
        #same as the process from the snowballEarf function
        spherecord_all = lamb*d_sec*np.matmul(B,T_all)*dAxz
        T_all += spherecord_all
        radiative = (1-albedo)*insol-emiss*sigma*(T_all+273)**4
        T_all += d_sec * radiative / (rho*c*mxdlyr)
        T_all = np.matmul(np.linalg.inv(L), T_all)
    
    #Plotting to show how close they really are!
    fig2, axes = plt.subplots(1, 1)
    axes.plot(lats, T_all, color='red', label ='Diff + SphCorr + Radiative')
    axes.plot(lats, T_warm_init, color='teal',label = 'Initial Condition')
    #A little note on the graph so you can see what the values are (to not get confused)
    axes.annotate('Emissivity: {} & Diffusivity: {}'.format(emiss,lamb), xy=(0.3,-0.22), 
                xycoords='axes fraction', fontsize=9)
    plt.title('Adjusting the Lamda and Emissivity')
    plt.xlabel('Latitude')
    plt.ylabel('Temperature $^\circ$C')
    plt.legend()
    
def snowballEarfSnow(npoints = 18, dt = 1, tstop = 10000, lamb=100, S0 = 1370, 
                       emiss = 1, albedo_ice = 0.6, albedo_gnd = 0.3, temp_surf = 0,
                       FlashFreeze = False, temperature = 0, dynamic = False,
                       solar_mult = False, gamma_original = 1):
    '''
    This is function with all the meat and potatoes for this lab. This answers 
    all varying condions for questions/steps 3 & 4
    

    Parameters
    ----------
    npoints : int, optional
        Same as the gen_grid, it helps to determine latitudes. The default is 18.
    dt : int, optional
        Change of time timestep. The default is 1 (year).
    tstop : int, optional
        The total amount of timesteps that will be taken. The default is 10000 (years).
    lamb : float, optional
        The diffusivity constant for the heat equation. The default is 100 (m^2/s).
    emiss : float, 0-1, optional
        The emissivity value of radiating surface of the planet. The default is 1.
    S0 : float, optional
        This is the solar constant, used for solar insolation factor. The default is 1370 W/m^2.
        
    #New Variables
    albedo_ice : float (0-1), optional
        This is the value for reflection of surfaces of the cold/ice variety. The default is 0.6.
    albedo_gnd : float (0-1), optional
        This is the value for reflection of surface for the warm/land/dirt variety. The default is 0.3.
    temp_surf : float, optional
        If you want a constant surface temperature around the globe, this variable will make 
        all latitudes that temperature value. The default is 0.
    FlashFreeze : bool, optional
        If set to true, it puts in the flashfreeze condtions of a completly frozen earth
        with an albedo of 0.6. The default is False.
    temperature : bool, optional
        This is a condtion to make an equal temperature planet. The difference between this
        and temp_surf is that temp_surf actually assigns the surface temperature value.
        The default is 0 (meaning not equal temperature over earth).
    dynamic : bool, optional
        This turns on the dynamic albedo properteries if turned to true. It just reassigns albedo
        if the temperature changes above or below -10C. The default is False.
    solar_mult : bool, optional
        This turns a solar multipler (insolation term) on if set to true (its just an added
        a factor to the solar insolation term) . The default is False.
    gamma_original : bool, optional
        The preps the gamama term before called on for the solar_mult parameter. 
        The default is 1 meaning it doesnt do anything. 0, the other option will
        put together two arrays to make a gamma going up and a gamma going down

    Returns
    -------
    T_change : float array
        The final temperature of each latitude point/ In an array of npoints in 
        length
    lats : floot array
        The latitudes used to create the T_change values. Also npoints in length
        Returned for plotting
    T_mult : float array
        This is the average temperatures over all laitudes when the solar_mult parameter is turned on.
        Will be the length of the gammas array
    gammas : float array
        The gamma values determined at the begining of the wrriten code (also hard coded in whoops)
        Returned for plotting

    '''
    if gamma_original == 0:
        
        gamma = np.arange(0.4,1.45,0.05)
        gamma_back = np.arange(1.35,0.35,-0.05)

        gammas = np.zeros(2*len(gamma)-1)
        
        gammas[0:len(gamma)] = gamma

        gammas[len(gamma):] = gamma_back

        T_mult = np.zeros(len(gammas))
        
    if gamma_original == 1:
        T_mult = gamma_original
        gammas = gamma_original

    
    sigma = 5.67*10**-8 
    rho =1020
    c = 4.2*10**6
    mxdlyr =50
    
    dlat = 10
    dlat, lats, edges = gen_grid(npoints)
    
    T_change=np.zeros(len(lats))  
        
    
    if temperature == 1:
        T_change = np.zeros(len(lats))
        T_change[:] = temp_surf
    
    if temperature == 0:
        T_change = sbf.temp_warm(lats)
    
    
    if FlashFreeze == True:
        albedo = albedo_ice
        #print("Here")
        
    if FlashFreeze == False:
        albedo = albedo_gnd
    
    
    #Set delta y
    dy = np.pi * radearth/ npoints

        

    #Tri-diagnal Set=up
    A = -2*np.identity(npoints)
    for i in range(1,len(A)-1):
            A[i,i+1] = 1
            A[i,i-1] = 1
    
    #Boundary Conditions
    A[0,1] = 2
    A[-1,-2] = 2
    
    d_sec = dt*60*60*24*365
    a_matrix = (1/(dy**2))* A
    L = np.identity(npoints) - (lamb*d_sec*a_matrix)
    
    #Central Difference Craziness
    B = np.zeros((npoints,npoints))
    B[np.arange(npoints-1), np.arange(npoints-1)+1] = 1
    B[np.arange(npoints-1)+1, np.arange(npoints-1)] = -1
    B[0,:] = B[-1,:] = 0
    
    Axz = np.pi*((radearth+50.0)**2 - radearth**2)*np.sin(np.pi/180. *lats)
    dAxz = np.matmul(B,Axz)/(Axz*4*dy**2)
   
    nsteps = int(tstop/dt)
    
    #Setting up Insolation:
    if solar_mult == False:
        insol = sbf.insolation(S0, lats)
        for i in range(nsteps):
            # Update albedo based on conditions:
            if dynamic == True :
                albedo = np.zeros(len(lats))
                loc_ice = T_change <= -10
                albedo[loc_ice] = albedo_ice
                albedo[~loc_ice] = albedo_gnd
            if dynamic == False:
                albedo = albedo
       
            spherecord_all = lamb*d_sec*np.matmul(B,T_change)*dAxz
            T_change += spherecord_all
            radiative = (1-albedo)*insol-emiss*sigma*(T_change+273)**4
            T_change += d_sec * radiative / (rho*c*mxdlyr)
            T_change = np.matmul(np.linalg.inv(L), T_change)
    
    if solar_mult == True:
        
        for j in range(len(gammas)): 
            insol = gammas[j] * sbf.insolation(S0, lats)
            for i in range(nsteps):

                if dynamic == True :
                    albedo = np.zeros(len(lats))
                    loc_ice = T_change <= -10
                    albedo[loc_ice] = albedo_ice
                    albedo[~loc_ice] = albedo_gnd
                if dynamic == False:
                    albedo = albedo
                    
                spherecord_all = lamb*d_sec*np.matmul(B,T_change)*dAxz
                T_change += spherecord_all
                radiative = (1-albedo)*insol-emiss*sigma*(T_change+273)**4
                T_change += d_sec * radiative / (rho*c*mxdlyr)
                T_change = np.matmul(np.linalg.inv(L), T_change)
            T_mult[j] = np.mean(T_change)
    
    
    return (T_change, lats, T_mult, gammas)
    
snowballEarf()

snowballEarfWarm(emiss=0.72, lamb = 50)

###############################Question 3######################################
twarm, lats, garg, who = snowballEarfSnow(emiss= 0.72, lamb = 50, temp_surf= 60, 
                                     dynamic= True, temperature = 1)
tcold, lats, garg, cares = snowballEarfSnow(emiss = 0.72, lamb = 50, temp_surf= -60, 
                                     dynamic = True, temperature = 1)
t_flash, lats, garg, notme = snowballEarfSnow(emiss = 0.72, lamb = 50, temperature = 0,
                                       FlashFreeze = True )
# Create a figure/axes object
fig, axes = plt.subplots(1, 1)

axes.plot(lats, tcold, label = 'Cold Temperatures')
axes.plot(lats, twarm, label = 'Warm Temperatures')
axes.plot(lats, t_flash, label = 'Flash-Freeze Scenario')
plt.title('Hot Earth VS Cold Earth')
plt.xlabel('Latitude')
plt.ylabel('Temperature $^\circ$C' )
plt.legend()
#plt.savefig('test.png')

################################Question 4#####################################

tgarg, lats, t_mult, gamma = snowballEarfSnow(npoints = 50, emiss= 0.72, lamb= 50, temp_surf= -60, 
                                     dynamic= True, temperature = 1, solar_mult= True,
                                     gamma_original= 0)


# Create a figure/axes object
fig, axes = plt.subplots(1, 1)

axes.plot(t_mult[0:20], gamma[0:20], label = 'Gamma Going Up')
axes.plot(t_mult[20:], gamma[20:], label = 'Gamma Going Down')

#axes.plot(gamma[0:20], t_mult[0:20], label = 'Gamma Going Up')
#axes.plot(gamma[20:],t_mult[20:], label = 'Gamma Going Down')

plt.title('Mean Global Temperature VS Gamma')
plt.xlabel('Mean Temperature $^\circ$C' )
plt.ylabel('Gamma Value')
plt.legend()






