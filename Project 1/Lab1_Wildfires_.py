#!/usr/bin/env python3
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np


def wildfire_spread(ny,nx,numiters,prob_spread,prob_bare,prob_start,plot_ident):

    '''
    Function Description
    ====================
    
    Takes in all the parameters to make a series of color coded plots that simulate the spread of a wildfire by close contact. 
    
    
    Parameters:
    =============
    ny - the total amount of grid spaces in the y direction (height/tall)
    
    nx - the total amount of grid spaces in the x direction (length/wide)
    
    numiters - the finite amount of iterations the function will run whether or not the fire has died out
    
    prob_spread - a number from 0-1 to indicate the proability of a fire square to spread to an adjacent forest cell
    
    prob_bare - a number from 0-1 to indicate the starting probablity of a forest square to be a bare cell
    
    prob_start - a number from 0-1 to indicate the starting probablity of a forest square to be a fire cell
    
    plot_ident - a way to distingish multiple sets of plots if the function is run consecutive times
                it can also be any data type, just needs to be indicated as such
    
    Returns:
    ========
    No direct returns in this function. The final product are the images (.png) in the figures folder
    that lay out the iterations in color
    
    '''
    
    
    
    # Create an initial grid, set all values to "2". dtype sets the value
    # type in our array to integers only.
    
    #initalizing some variables
    
    forest  =  np.zeros([numiters,ny,nx], dtype=int) + 2
    information = []
    
    #1 is bare
    #2 is forest
    #3 is on fire
    
    #sets a random amount of grid squares to start as bare
    for i in range(nx):
        # Loop in the "y" direction:
        for j in range(ny):
            ## Perform logic here:
            if np.random.rand() < prob_bare:
                forest[0,j, i] = 1
    
    # Set a certain amount of grid squares to start as burning:
    for i in range(nx):
        # Loop in the "y" direction:
        for j in range(ny):
            ## Perform logic here:
            if np.random.rand() < prob_start:
                forest[0,j, i] = 3
            ##Case when the fire needs to start in the middle of the grid
            if prob_start == 0.0:
                a = (ny-1)/2
                #print(int(a))
                b = (nx-1)/2
                forest[0,int(a),int(b)] = 3
                
                
                
                
    
    #Specifc color code for the plots to show the difference in bare,forest,and fire
    forest_cmap = ListedColormap(['tan', 'darkgreen', 'crimson'])
    
    #Plot for the starting conditions for the process
    fig, ax = plt.subplots(1,1)
    ax.pcolor(forest[0,:,:], cmap=forest_cmap, vmin=1, vmax=3)
    
    #Easier readability
    ax.set_title('{} Wild Fire Burn Starting Iteration'.format(plot_ident))
    ax.set_xlabel('X units')
    ax.set_ylabel('Y units')
    
    #To easily access the photos. I created a folder of figures in my working directory, may need to add a similar file to yours
    fig.savefig(('figures/wildfire/{} Starting Iteration.png').format(plot_ident))
     
    
    ##########################################################################
    
    #Main fire spread loop
    for k in range(1, numiters):
        forest[k,:,:] = forest[k-1,:,:]
        
        for i in range(nx):
            for j in range(ny):
            #starting to go in all the directions for fire spreading for one particular point
            
                if forest[k-1,j,i]==3:
                    
                    #going left, if not on the left most edge and is a forest cell, fire could spread
                    if i!=0 and forest[k-1,j,i-1]==2:
                        firespread = np.random.rand()
                        if firespread  <= prob_spread:
                            forest[k,j, i-1] = 3
    
                    #going down, if not on the bottom most edge and is a forest cell, fire could spread
                    if j!=0 and forest[k-1,j-1,i]==2:
                        firespread = np.random.rand()
                        if firespread  <= prob_spread:
                            forest[k,j-1,i] = 3
                        
                    #going right, if not on the right most edge and is a forest cell, fire could spread
                    if i!=nx-1 and forest[k-1,j,i+1]==2:
                        firespread = np.random.rand()
                        if firespread  <= prob_spread:
                            forest[k,j, i+1] = 3
                        
                    #going up, if not on the top most edge and is a forest cell, fire could spread
                    if j!=ny-1 and forest[k-1,j+1,i]==2:
                        firespread = np.random.rand()
                        if firespread  <= prob_spread:
                            forest[k,j+1, i] = 3
                    
                    #current fire cell becomes bare
                    forest[k,j,i] = 1
        
        
        
        
        
        #For creating the plots for each iteration
        fig, ax = plt.subplots(1,1)
        ax.pcolor(forest[k,:,:], cmap=forest_cmap, vmin=1, vmax=3)
        ax.set_title('{} Wild Fire Burn Iteration: {}'.format(plot_ident, k))
        ax.set_xlabel('X units')
        ax.set_ylabel('Y units')
        fig.savefig('figures/wildfire/{} Iteration-{}'.format(plot_ident, k))
        plt.close()
        
        
        
        
        #Collecting data for experimenting
            #Grabing total iterations taken
            #Amount of each category (burning,forest,bare) through each iteration
        category_info = [np.sum(forest[k,:,:]==1),np.sum(forest[k,:,:]==2),np.sum(forest[k,:,:]==3)]
        information.append(category_info)
    
        
        #to stop making graphs when there is no more fire left to burn anything
        if np.sum(forest[k,:,:]==3) == 0:
            break
    
    
    
    
    #Set-up for the plotting
    
    #Creating figures with the information collected about the cases
    fig, ax = plt.subplots(1,1)
    ax.plot(range(0,len(information)),[i[0] for i in information], label = 'Bare Points')
    ax.plot(range(0,len(information)),[i[1] for i in information], label = 'Forest Points')
    ax.plot(range(0,len(information)),[i[2] for i in information], label = 'Burning Points')
    ax.set_title('{} Change in Category of Gridpoints After Each Iteration'.format(plot_ident))
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Gridpoints')
    ax.legend()
    fig.savefig('figures/wildfire/{} Category Plot'.format(plot_ident))
    plt.close()



def disease_spread(ny,nx,numiters,prob_fatal,prob_spread,prob_immune,prob_sick,plot_ident):

    '''
    Function Description
    ====================
    Takes in all the parameters to make a series of color coded plots that simulate the spread of a disease by close contact. 
    
    
    Parameters:
    =============
    ny - the total amount of grid spaces in the y direction (height/tall)
    
    nx - the total amount of grid spaces in the x direction (length/wide)
    
    numiters - the finite amount of iterations the function will run whether or not there is no more sickness
    
    prob_fatal - a number from 0-1 to indicate the probability of a sick person dying from the disease
    
    prob_spread - a number from 0-1 to indicate the probability of a sick person spreading it to an adjacent person (gridpoint)
    
    prob_immune - a number from 0-1 to indicate the starting probability of people that have taken the vaccine and can not get sick
    
    prob_sick - a number from 0-1 to indicate the starting probability of people that are sick
    
    plot_ident - a way to distingish multiple sets of plots if the function is run consecutive times
                it can also be any data type, just needs to be indicated as such
    
    Returns:
    ========
    No direct returns in this function. The final product are the images (.png) in the figures folder
    that lay out the iterations in color
    
    '''

    # Create an initial grid, set all values to "2". dtype sets the value
    # type in our array to integers only.
    #0 is dead - brown
    #1 is immune - azure
    #2 is healthy - orange
    #3 is sick - green
    
    #initalizing some variables
    population  =  np.zeros([numiters,ny,nx], dtype=int) + 2
    information = []
     
    #sets a random amount of grid squares to start as immune
    for i in range(nx):
        # Loop in the "y" direction:
        for j in range(ny):
            ## Perform logic here:
            if np.random.rand() < prob_immune:
                population[0,j, i] = 1
     
    # Set a certain amount of grid squares to start as sick:
    for i in range(nx):
        # Loop in the "y" direction:
        for j in range(ny):
            ## Perform logic here:
            if np.random.rand() < prob_sick:
                population[0,j, i] = 3
            ##Case when the sickness needs to start in the middle of the grid
            if prob_sick == 0.0:
                a = (ny-1)/2
                b = (nx-1)/2
                population[0,int(a),int(b)] = 3
                
                
                
                
     
    #Specifc color code for the plots to show the difference in dead, immune, healthy, and sick
    population_cmap = ListedColormap(['brown','azure', 'orange', 'green'])
     
    #Plot for the starting conditions for the process
    fig, ax = plt.subplots(1,1)
    ax.pcolor(population[0,:,:], cmap=population_cmap, vmin=0, vmax=3)
     
    #Easier readability
    ax.set_title('{} Disease Population Starting Iteration'.format(plot_ident))
    ax.set_xlabel('X units')
    ax.set_ylabel('Y units')
     
    #To easily access the photos. I created a folder of figures in my working directory, may need to add a similar file to yours
    fig.savefig(('figures/disease/{} Starting Iteration.png').format(plot_ident))

   

    
    ##########################################################################
    
    #Main disease spread loop
    for k in range(1, numiters):
        population[k,:,:] = population[k-1,:,:]
        
        for i in range(nx):
            for j in range(ny):
            #starting to go in all the directions for fire spreading for one particular point
            
                if population[k-1,j,i]==3:
                    
                    #going left, if not on the left most edge and is a population cell, sickness could spread
                    if i!=0 and population[k-1,j,i-1]==2:
                        sickspread = np.random.rand()
                        if sickspread  <= prob_spread:
                            population[k,j, i-1] = 3
    
                    #going down, if not on the bottom most edge and is a population cell, sickness could spread
                    if j!=0 and population[k-1,j-1,i]==2:
                        sickspread = np.random.rand()
                        if sickspread  <= prob_spread:
                            population[k,j-1,i] = 3
                        
                    #going right, if not on the right most edge and is a population cell, sickness could spread
                    if i!=nx-1 and population[k-1,j,i+1]==2:
                        sickspread = np.random.rand()
                        if sickspread  <= prob_spread:
                            population[k,j, i+1] = 3
                        
                    #going up, if not on the top most edge and is a population cell, sickness could spread
                    if j!=ny-1 and population[k-1,j+1,i]==2:
                        sickspread = np.random.rand()
                        if sickspread  <= prob_spread:
                            population[k,j+1, i] = 3
                    
                    #current fire cell now has an option to be fatal or make the person immune
                    fatal = np.random.rand()
                    if fatal <= prob_fatal:
                        population[k,j,i] = 0
                    else:
                        population[k,j,i] = 1
                        
                        
                        
                        
        
        #For creating the plots for each iteration
        fig, ax = plt.subplots(1,1)
        ax.pcolor(population[k,:,:], cmap=population_cmap, vmin=0, vmax=3)
        ax.set_title('{} Population Disease Spread Iteration: {}'.format(plot_ident,k))
        ax.set_xlabel('X units')
        ax.set_ylabel('Y units')
        fig.savefig('figures/disease/{} Iteration-{}'.format(plot_ident,k))
        plt.close()
        
        
        
        
        #Collecting data for experimenting
            #Grabing total iterations taken
            #Amount of each category (dead, immune, healthy, sick) through each iteration
        category_info = [np.sum(population[k,:,:]==0),np.sum(population[k,:,:]==1),np.sum(population[k,:,:]==2),np.sum(population[k,:,:]==3)]
        information.append(category_info)
        
        #to stop making graphs when there is no more fire left to burn anything more
        if np.sum(population[k,:,:]==3) == 0:
            break
        
        
        
        
    
    #Set-up for the plotting
    
    #Creating figures with the information collected about the cases
    fig, ax = plt.subplots(1,1)
    ax.plot(range(0,len(information)),[i[0] for i in information], label = 'Dead People')
    ax.plot(range(0,len(information)),[i[1] for i in information], label = 'Immune People')
    ax.plot(range(0,len(information)),[i[2] for i in information], label = 'Healthy People')
    ax.plot(range(0,len(information)),[i[3] for i in information], label = 'Sick People')
    #ax.plot(range(0,len(information)),[i[4] for i in information], label = 'Surviors')
    ax.set_title('{} Change in Category of Population After Each Iteration '.format(plot_ident))
    ax.set_xlabel('Iterations')
    ax.set_ylabel('People')
    ax.legend()
    fig.savefig('figures/disease/{} Category Plot'.format(plot_ident))
    plt.close()


'''
Running the function(s)
==============================================================================
Below are all the commands grouped to the questions or parts of questions that
they are concerned with
 
'''
#Running wildfire function for different scenarios

#Testcase with a 3x3 grid, 100% spread, no starting bare, middle square on fire
wildfire_spread(3,3,4,1,0,0,'Testcase')

#Testcase where the grid is wider than it is tall
wildfire_spread(25,45,40,1,0.0,0.0,'Wider')

#Cases varying prob_spread
wildfire_spread(100,100,50,0.2,0.0,0.1,'Slow Spread')
wildfire_spread(100,100,50,0.5,0.0,0.1,'Coin-flip Spread')
wildfire_spread(100,100,50,0.8,0.0,0.1,'Fast Spread')

#Cases varying prob_bare
wildfire_spread(100,100,50,0.5,0.05,0.1,'Small Bare')
wildfire_spread(100,100,50,0.5,0.15,0.1,'Medium Bare')
wildfire_spread(100,100,50,0.5,0.3,0.1,'Large Bare')



#Running the disease function for different scenarios

#Cases varying in prob_fatal
disease_spread(100,100,50,0.2,0.5,0.0,0.1,'Low Fatal')
disease_spread(100,100,50,0.5,0.5,0.0,0.1,'Medium Fatal')
disease_spread(100,100,50,0.8,0.5,0.0,0.1,'High Fatal')


#Cases varying in prob_immune
disease_spread(100,100,50,0.3,0.5,0.05,0.1,'Small Immune')
disease_spread(100,100,50,0.3,0.5,0.15,0.1,'Medium Immune')
disease_spread(100,100,50,0.3,0.5,0.3,0.1,'Large Immune')



   
























