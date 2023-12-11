#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 18:51:01 2023

@author: torischeidt
"""
#import necessary packages
import numpy as np
import matplotlib.pyplot as plt

#Setting up some constants
g = 9.81 #m/s^2
R = 6373 #km
water_density = 1025 #kg/m^3

#Setting coordinates easier for changing later
ocean_lat_pac = 31.430333
ocean_lon_pac = -177.902686

japan_lat = 35.191124
japan_lon = 140.401974

ocean_lat_atl = 34.852427
ocean_lon_atl = -40.809877

newyork_lat = 40.579363
newyork_lon = -74.036040

indian_coast_lat = 13.080270
indian_coast_lon = 80.294804

sumatra_coast_lat = -2.028408
sumatra_coast_lon = 97.274088

#Implementing the short water equation
def tsunami(H, wavelength, distance = 0, latlon = False, lat1 = 0, lon1=0, lat2=1, lon2=1):
    '''
    This function calculates the shallow water equation with a couple extra perks including
    ocean depth to wavelength limit and the latlon coordinate to distance conversion

    Parameters
    ----------
    H : Float (in km)
        This is the ocean depth of water. Usually shouldnt go above 6,000 meters
    wavelength : Float (in km)
        This is the wavelength of the wave, which is typically pretty long
    distance : Float, optional (in km)
        If you are just wanting to do a distance the wave travels. The default is 0.
    latlon : Bool, optional
        This will trigger the if statement to do the calculations for the latlon coversion The default is False.
    lat1 : Float, optional (in degrees)
        Latitude coordinate of point 1. The default is 0.
    lon1 : Float, optional  (in degrees)
        Longitude coordinate of point 1 The default is 0.
    lat2 : Float, optional  (in degrees)
        Latitude coordinate of point 2. The default is 1.
    lon2 : Float, optional  (in degrees)
        Longitude coordinate of point 2. The default is 1.

    Raises
    ------
    Exception
        This tests if the shallow water equation applies. If it does not, it will stop the running the code

    Returns
    -------
    time : Int (in seconds)
        The time it takes for the wave to travel over the distance.
    wavelength : Float (in m)
        This is just the wavelength used for the exception. Really doesnt need to be pulled through but
        it makes it easier to keep the same one when you are doing the wave plot. You only have to change
        it once then
    distance : Float (in km)
        This is the total distance that the wave travels 
    c : Float (in m/s)
        The phase speed of the wave and the main reason for this function
    c_kmhr : Float (in km/hr)
        The phase speed but converted for better comparisons 

    '''
    #Coverting things into meters. It is easier to type km into the function
    H = H*1000
    wavelength = wavelength*1000
    
    #Limit for shallow water equation
    if H >= 0.05*wavelength:
        raise Exception('Shallow water equation does not apply, please try a different Ocean Depth value')
        
    #Shallow Water Approx (in m/s)
    c = (H*g)**(1/2)
    
    #Phase speed converted into km/hr
    c_kmhr = c*3600/(1000)
    
    #Time it will take for wave to get there
    time = (distance*1000)/c
    
    if latlon == True:
        
        #Converting all coordinates into radians for the proper calculations
        lat1 = np.radians(lat1)
        lon1 = np.radians(lon1)
        lat2 = np.radians(lat2)
        lon2 = np.radians(lon2)
        
        #Calculating the change in longitude and latitudes
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        
        #The actual magical equations to get distance
        a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
        p = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        distance = R * p
        
        #Time it will take for wave to get there
        time = (distance*1000)/c
        
    return time, wavelength, distance, c, c_kmhr

def energy(wave_height, surface_area = 0):
    '''
    This function calculates the energy and energy per square meter of a wavr

    Parameters
    ----------
    wave_height : Float (in m)
        This is the final height that a wave gets to be when it approaches the shore
    surface_area : Float, optional (in m^2)
        If you have the surface area of the coast/beach you are trying to calulate energy. 
        The default is 0.

    Returns
    -------
    energy_area : Float (rounded)
        The amount of energy per square meter
    energy : Float (rounded)
        The amound of energy over a specfic area

    '''
    energy = 0
    
    #Total Energy per unit of sea surface area
    energy_area = round((1/8 * water_density * g * wave_height**2),4)
    
    #Surface area in m**2
    if not surface_area == 0:
        energy = round((energy_area*surface_area),4)
    
    return energy_area, energy

#Plotting the wave on a graph to visually see the difference in waves     
def wave(time, a, wavelength, c, distance, dx=1000, dt = 1000, energy_calc = False, wave_height = 2, surface_area = 0):
    '''
    This equation is for plotting the tsunami wave using the wave equation 

    Parameters
    ----------
    time : Int (s)
        The amount of time it takes for the tsunami wave to go the distance 
    a : Float (0-1 m)
        Amplitude of the wave in open water. This variable is used in the exception
    wavelength : Float (m)
        Wavelength of the tsunami wave
    c : Float (m/s)
        Phase speed of the wave
    distance : Float (km)
        Distance that the wave travels between two locations
    dx : Int, optional
        The spacial step to create the plot. The default is 1000.
    dt : Int, optional
        The time step to create the plot. The default is 1000.
    
    #For the energy calculations
    energy_calc : Bool, optional
        Triggers the energy calculations inside the if statement. The default is False.
    wave_height : Float (in m)
        This is the final height that a wave gets to be when it approaches the shore
    surface_area : Float, optional (in m^2)
        If you have the surface area of the coast/beach you are trying to calulate energy. 
        The default is 0.

    Raises
    ------
    Exception
        To check that the amplitude is not too big for a tsunami wave. Also makes the
        plotting more uniform for better visual comparisons

    Returns
    -------
    None.

    '''
    #For plotting the actual wave postion over the ocean, good visual
    
    if a > 1:
        raise Exception('Not a valid amplitude. Please choose a value less than 1 meter')
    
    #Making a timestep array & distance array 
    time_array = np.linspace(0,time,dt)
    dist_array = np.linspace(0,distance,dx)
    
    #Setting up the height array 
    height_y = np.zeros(len(time_array))
    
    for i in range(0,len(time_array)):
        height_y[i] = a*np.sin((2*np.pi/wavelength)*(dist_array[i]-c*1000*time_array[i])) 
    
    time_hr = time/(60*60)
    
    #Plotting the wave 
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.plot(dist_array, height_y)
    
    #Formatting
    ax.set_title('Tsunami Wave Over Distance {} km & Duration {} hours'.format(
        round(distance,3),round(time_hr,2)),fontsize=17)
    ax.set_xlabel('Distance Over Water (km)', fontsize=15)
    ax.set_ylabel('Height of Wave (m)', fontsize=15)
    ax.set_ylim(-1,1)
    
    if energy_calc == True:
       energy_area_T, energy_T = energy(wave_height, surface_area)
       ax.annotate('Energy (J) per m$^2$ = {}'.format(energy_area_T), 
                   xy=(0.67,-0.25), xycoords='axes fraction', fontsize=12)
       if not surface_area == 0:
           ax.annotate('Energy (J) = {}'.format(energy_T), 
                       xy=(0,-0.25), xycoords='axes fraction', fontsize=12)

#################################Question 1####################################
time_T, wavelength_T, distance_T, phase_speed, phase_speedkmhr = tsunami(6, 400, distance = 3000)
wave(time_T, 0.5, wavelength_T, phase_speed, distance_T)

#################################Question 3####################################
#Doing something over the atlantic
time_T, wavelength_T, distance_T, phase_speed, phase_speedkmhr = tsunami(3.646, 400, latlon= True,
                                                                         lat1 = ocean_lat_atl, lon1 = ocean_lon_atl,
                                                                         lat2 = newyork_lat, lon2 = newyork_lon)
wave(time_T, 0.5, wavelength_T, phase_speed, distance_T)
print(phase_speed)


#Doing something over the Pacific 
time_T, wavelength_T, distance_T, phase_speed, phase_speedkmhr = tsunami(4.28, 400, latlon= True,
                                                                         lat1 = ocean_lat_pac, lon1 = ocean_lon_pac,
                                                                         lat2 = japan_lat, lon2 = japan_lon)
wave(time_T, 0.5, wavelength_T, phase_speed, distance_T)
print(phase_speed)
#################################Question 4####################################
#Check against Sumatra earthquake on Dec 26, 2004

time_T, wavelength_T, distance_T, phase_speed, phase_speedkmhr = tsunami(5, 550, latlon= True,
                                                                         lat1 = sumatra_coast_lat, lon1 = sumatra_coast_lon,
                                                                         lat2 = indian_coast_lat, lon2 = indian_coast_lon)

wave(time_T, 0.7, wavelength_T, phase_speed, distance_T, energy_calc= True, wave_height = 50)
print(phase_speed)


