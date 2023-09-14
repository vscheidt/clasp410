#!/usr/bin/env python3
'''
This file contains tools and scripts for completing Lab 1 for CLaSP 410.
To reproduce the plots shown in the lab report, do this...
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

forest_cmap = ListedColormap(['tan', 'darkgreen', 'crimson'])

#1 is bare
#2 is forest
#3 is on fire

nx, ny, numiters = 3, 3, 3 # Number of cells in X and Y direction.
prob_spread = 1.0 # Chance to spread to adjacent cells.
prob_bare = 0.0 # Chance of cell to start as bare patch.
prob_start = 0.0 # Chance of cell to start on fire.


# Create an initial grid, set all values to "2". dtype sets the value
# type in our array to integers only.
forest  =  np.zeros([numiters,ny,nx], dtype=int) + 2


#Random staring number of bare
isbare = np.random.rand(ny, nx)
isbare = isbare < prob_bare
forest[0,isbare] = 1
            
# Set the center cell to "burning":
forest [1,1] = 3

#fire spread loop
for k in range(1, numiters):
    #making sure all the data is being transfered over before a new iteration
    forest[k,:,:] = forest[k-1,:,:]
    for i in range(ny):
        for j in range(nx):
            
            if forest[k,i,j] == 3:
                if forest[k,i,j-1]==2:
                    forest[k,i,j-1] = 3
                if forest
                
            
            if
            
            if
            # Roll our "dice" to see if we get a bare spot:
            if forest[k-1,j,i]==3:
                if forest[k-1,j,i-1]==2:
                    forest[j, i-1] = 3



plt.ioff()
fig, ax = plt.subplots(1,1)
ax.pcolor(forest, cmap=forest_cmap, vmin=1, vmax=3)






















