#!/usr/bin/env python3

'''
This is the solution for CLaSP 410 Lab 1: Forest Fires and Diseases.

The forest is represented by a 3x3 grid. Each cell has a status:
    1: Bare
    2: Forest
    3: Actively Burning
    4: Deceased (disease model only.)

Fire can spread to orthogonal neighbors only given some probability of spread.
At each time iteration, we evaluate each cell. If it's burning, we determine
if the fire spreads to each neighbor. If it does, and the neighbor's status
is "Forest", it is changed to "Recently Ignited".
After the spreading step, all "Actively Burning" cells are switched to "Bare"
cells and "Recently Ignited" cells are switched to "Actively Burning" cells.

The main tool in this module for simulating forest fires is the `forest_fire`
function. Read the documentation for that to see how it works.

Teaching Strategy:
We will walk through this solution slowly together.
    First pass: create simple script version of `forest_fire` with no
    functions. Use brute-force loops over all arrays. Plotting functions are
    hard-coded and dump images at every iteration. Use extra "4th" version.
    2D array of forest only. This approach will be used to create our 3x3
    validation result.

    Second pass: Create functions to produce a simple module for forest fire
    modeling (but don't worry about imports quite yet!). Main function solves
    for forest fire given input parameters of size, burn rate, etc.
    Forest array is now 3D with 3rd dimension being time. This 3D array becomes
    the only return variable for the forest fire. All plotting and processing
    are done by other functions.

    Third pass: Reduce brute-force loops with array/vector operations. Add an
    option to stop the simulation if there is no change in solution.

'''

import os

import numpy as np
import matplotlib.pyplot as plt

# We will use this to do the colorbars on our 2D plots.
from matplotlib.colors import ListedColormap

# Set default style:
plt.style.use('fivethirtyeight')

# Generate our custom segmented color map for this project.
# We can specify colors by names and then create a colormap that only uses
# those names. We have 3 funadmental states, so we want only 3 colors.
# Color info: https://matplotlib.org/stable/gallery/color/named_colors.html
colors = ['tan', 'darkgreen', 'crimson']
forest_cmap = ListedColormap(colors)

# Create a dictionary to turn status into words:
status = {1: 'Bare', 2: 'Forest', 3: 'FIRE!'}

# Repeat the above, but now create a set for disease spread:
colors_sick = ['black', 'dodgerblue', 'orange', 'crimson']
forest_sick = ListedColormap(colors_sick)
status_sick = {0: 'Dead', 1: 'Immune', 2: 'Healthy', 3: 'Infected'}


def forest_fire_slow(nx=3, ny=3, nstep=3, prob_bare=0.0, prob_spread=1.0,
                     prob_start=0.0, prob_fatal=0.0, ignite=False):
    '''
    THIS FUNCTION IS THE SAME AS THE ONE BELOW, BUT USES A BRUTE-FORCE
    LOOPING APPROACH INSTEAD OF A MORE ELEGANT LOGICAL INDEXING METHOD.

    This function intiates and executes a simple forest fire model.
    This function returns the `forest` array of shape [niter, ny, nx] where
    the integer value of each cell indicates the status of that chunk of
    forest at some iteration.

    The value of any cell can be one of the following:
        1 (burnt/sick and recovered)
        2 (Forested/healthy person)
        3 (Burning/sick person)
        4 (Deceased)

    Parameters
    ===========
    nx, ny : int, defaults to 3
        Size of grid in X and Y directions, respectively.
    nstep : int, defaults to 3
        Maximum number of steps to take; defaults to 3.
    prob_bare : float, defaults to 0
        Probability of each cell starting as a bare spot.
    prob_spread : float, defaults to 1.0
        Probability fire will spread between cells for each iteration.
    prob_start : float, defaults to 0
        Probability of each cell starting on fire.
        Use this to randomly seed fire across domain.
    ignite : False or list of lists, defaults to False
        Set of cells that begin on fire. For example, ignite=[[1,1], [1,2]]
        will set cells at x=1, y=1 and x=1, y=2 on fire at simulation start.
        Use this to explicitly set where fire will start.

    Returns
    =======
    forest : Numpy array.
        A 3D array representing the status of the forest whose dimensions are
        X, Y, and iteration.
    '''

    # Grab the random number generator function.
    from numpy.random import rand

    # Initialize grid to correct size. Set to all forest.
    forest = np.zeros((nstep, ny, nx), dtype=int) + 2

    # Brute force: loop over all values...
    for i in range(nx):
        for j in range(ny):
            # True * 1 = 1, False * 1 = 0. Hence, this line sets the status
            # to 1 (bare) if prob_bare is larger than the random roll.
            forest[0, j, i] = 2 - 1 * (rand() <= prob_bare)

    # Hard set ignition points:
    if ignite:
        for x, y in ignite:
            forest[0, y, x] = 3

    # Random ignition points:
    start = np.random.rand(ny, nx) < prob_start
    forest[0, start] = 3

    # Now, run the simulation:
    for istep in range(1, nstep):
        # Copy values from previous time step to current one:
        forest[istep, :, :] = forest[istep-1, :, :]

        # LOOP IMPLEMENTATION: This is the brute-force approach.
        # Spread fire from left to right (rightmost cells won't spread)
        for i in range(nx-1):
            for j in range(ny):
                # If this cell was not burning, keep going.
                if forest[istep-1, j, i] != 3:
                    continue
                # If it was burning, spread fire in each direction ONLY
                # if there is forest there to burn:
                if forest[istep, j, i+1] == 2:
                    # Roll dice. If less than prob_spread,
                    # change status from 2 to 1
                    forest[istep, j, i+1] += 1 * (rand() <= prob_spread)

        # Spread fire from right to left (leftmost cells won't spread).
        # Same as above, but spreading from i to i-1:
        for i in range(1, nx):
            for j in range(ny):
                # If this cell was not burning, keep going.
                if forest[istep-1, j, i] != 3:
                    continue
                # If it was burning, spread fire in each direction ONLY
                # if there is forest there to burn:
                if forest[istep, j, i-1] == 2:
                    # Roll dice. If less than prob_spread,
                    # change status from 2 to 1
                    forest[istep, j, i-1] += 1 * (rand() <= prob_spread)

        # Spread fire from bottom to top (top-most cells won't spread).
        # Same as above, but spreading from j to j+1:
        for i in range(nx):
            for j in range(ny-1):
                # If this cell was not burning, keep going.
                if forest[istep-1, j, i] != 3:
                    continue
                # If it was burning, spread fire in each direction ONLY
                # if there is forest there to burn:
                if forest[istep, j+1, i] == 2:
                    # Roll dice. If less than prob_spread,
                    # change status from 2 to 1
                    forest[istep, j+1, i] += 1 * (rand() <= prob_spread)

        # Spread fire from top to bottom (bottom-most cells won't spread)
        # Same as above, but spreading from j to j-1:
        for i in range(nx):
            for j in range(1, ny):
                # If this cell was not burning, keep going.
                if forest[istep-1, j, i] != 3:
                    continue
                # If it was burning, spread fire in each direction ONLY
                # if there is forest there to burn:
                if forest[istep, j-1, i] == 2:
                    # Roll dice. If less than prob_spread,
                    # change status from 2 to 1
                    forest[istep, j-1, i] += 1 * (rand() <= prob_spread)

        # Regions that were burning must now become bare.
        # Find those regions from the previous step:
        loc_burnt = forest[istep-1, :, :] == 3
        forest[istep, loc_burnt] = 1

        # Set "dead" (for disease spread only)
        loc_dead = loc_burnt * (np.random.rand(ny, nx) < prob_fatal)
        forest[istep, loc_dead] = 0

        # If no cells are burning, stop the simulation.
        if 3 not in forest[istep, :, :]:
            # Trim the output array down to minimal size.
            forest = forest[0:istep+1, :, :]
            # Break out of the loop.
            break

    return forest


def forest_fire(nx=3, ny=3, nstep=3, prob_bare=0.0, prob_spread=1.0,
                prob_start=0.0, prob_fatal=0.0, ignite=False):
    '''
    This function intiates and executes a simple forest fire model.
    This function returns the `forest` array of shape [niter, ny, nx] where
    the integer value of each cell indicates the status of that chunk of
    forest at some iteration.

    The value of any cell can be one of the following:
        1 (burnt/sick and recovered)
        2 (Forested/healthy person)
        3 (Burning/sick person)
        4 (Deceased)

    Parameters
    ===========
    nx, ny : int, defaults to 3
        Size of grid in X and Y directions, respectively.
    nstep : int, defaults to 3
        Maximum number of steps to take; defaults to 3.
    prob_bare : float, defaults to 0
        Probability of each cell starting as a bare spot.
    prob_spread : float, defaults to 1.0
        Probability fire will spread between cells for each iteration.
    prob_start : float, defaults to 0
        Probability of each cell starting on fire.
        Use this to randomly seed fire across domain.
    ignite : False or list of lists, defaults to False
        Set of cells that begin on fire. For example, ignite=[[1,1], [1,2]]
        will set cells at x=1, y=1 and x=1, y=2 on fire at simulation start.
        Use this to explicitly set where fire will start.

    Returns
    =======
    forest : Numpy array.
        A 3D array representing the status of the forest whose dimensions are
        X, Y, and iteration.
    '''

    # Initialize grid to correct size. Set to all forest.
    forest = np.zeros((nstep, ny, nx), dtype=int) + 2

    # Set bare patches:
    # Elegant approach: Numpy and logical indexing:
    bare = np.random.rand(ny, nx)
    forest[0, bare <= prob_bare] = 1

    # Hard set ignition points:
    if ignite:
        for x, y in ignite:
            forest[0, y, x] = 3

    # Random ignition points:
    start = np.random.rand(ny, nx) < prob_start
    forest[0, start] = 3

    # Now, run the simulation:
    for istep in range(1, nstep):
        # Copy values from previous time step to current one:
        forest[istep, :, :] = forest[istep-1, :, :]

        # ARRAY IMPLEMENTATION: These lines can replace all of the loops below.
        # Get locations of what is burning now and what has trees now:
        loc_burn = forest[istep, :, :] == 3
        loc_tree = forest[istep, :, :] == 2

        # Create probability matrix (one dice roll per grid point):
        ignite = np.random.rand(ny, nx) <= prob_spread

        # Spread fire. "doburn" has all the logic to indicate if
        # fire should spread or not (chance to ignite, neighbor has trees, etc)
        # Note the use of indexing: 1: is all but first cell, :-1 is all but
        # last cell. This allows us to avoid boundary issues.
        # Spread fire to right:
        doburn = loc_tree[:, 1:] * ignite[:, 1:] * loc_burn[:, :-1]
        forest[istep, :, 1:] = 3*doburn + ~doburn*forest[istep, :, 1:]
        # Spread fire to the left:
        doburn = loc_tree[:, :-1] * ignite[:, :-1] * loc_burn[:, 1:]
        forest[istep, :, :-1] = 3*doburn + ~doburn*forest[istep, :, :-1]
        # Spread fire upwards:
        doburn = loc_tree[1:, :] * ignite[1:, :] * loc_burn[:-1, :]
        forest[istep, 1:, :] = 3*doburn + ~doburn*forest[istep, 1:, :]
        # Spread fire downards:
        doburn = loc_tree[:-1, :] * ignite[:-1, :] * loc_burn[1:, :]
        forest[istep, :-1, :] = 3*doburn + ~doburn*forest[istep, :-1, :]

        # Regions that were burning must now become bare.
        # Find those regions from the previous step:
        loc_burnt = forest[istep-1, :, :] == 3
        forest[istep, loc_burnt] = 1

        # Set "dead" (for disease spread only)
        loc_dead = loc_burnt * (np.random.rand(ny, nx) < prob_fatal)
        forest[istep, loc_dead] = 0

        # If no cells are burning, stop the simulation.
        if 3 not in forest[istep, :, :]:
            forest = forest[0:istep+1, :, :]
            break

    return forest


def calc_percents(forest):
    '''
    Given a forest array as created by `forest_fire()`, calculate the
    percent of the forest that is bare, forested, and burning as a function
    of iteration.

    Three numpy arrays are returned: `per_bare`, `per_tree`, and `per_burn`.
    Each will be the same size as `forest.shape[0]` (i.e., same number of
    iterations as the incoming `forest` object).
    '''

    nstep = forest.shape[0]  # Number of iterations in simulation.
    npoints = forest.size / nstep  # Number of points in grid.
    per_bare = np.zeros(nstep)
    per_tree = np.zeros(nstep)
    per_burn = np.zeros(nstep)
    per_dead = np.zeros(nstep)

    # The loop way to do this:
    for i in range(nstep):
        per_bare = 100.0 * np.sum(1 * forest[i, :, :] == 1) / npoints
        per_tree = 100.0 * np.sum(1 * forest[i, :, :] == 2) / npoints
        per_burn = 100.0 * np.sum(1 * forest[i, :, :] == 3) / npoints
        per_dead = 100.0 * np.sum(1 * forest[i, :, :] == 0) / npoints

    # The array-way to do this:
    per_bare = 100.0 * np.sum(1 * forest == 1, axis=(1, 2)) / npoints
    per_tree = 100.0 * np.sum(1 * forest == 2, axis=(1, 2)) / npoints
    per_burn = 100.0 * np.sum(1 * forest == 3, axis=(1, 2)) / npoints
    per_dead = 100.0 * np.sum(1 * forest == 0, axis=(1, 2)) / npoints

    # Return either 3 or 4 arrays depending on if "dead" status was found.
    if sum(per_dead) > 0:
        return per_bare, per_tree, per_burn, per_dead
    else:
        return per_bare, per_tree, per_burn


def calc_percents_disease(forest):
    '''
    Given a forest array as created by `forest_fire()`, calculate the
    percent of the forest that is bare, forested, and burning as a function
    of iteration.

    Three numpy arrays are returned: `per_bare`, `per_tree`, `per_burn`, and
    `per_dead`.
    Each will be the same size as `forest.shape[0]` (i.e., same number of
    iterations as the incoming `forest` object).
    '''

    nstep = forest.shape[0]  # Number of iterations in simulation.
    npoints = forest.size / nstep  # Number of points in grid.
    per_bare = np.zeros(nstep)
    per_tree = np.zeros(nstep)
    per_burn = np.zeros(nstep)
    per_dead = np.zeros(nstep)

    # The array-way to do this:
    per_bare = 100.0 * np.sum(1 * forest == 1, axis=(1, 2)) / npoints
    per_tree = 100.0 * np.sum(1 * forest == 2, axis=(1, 2)) / npoints
    per_burn = 100.0 * np.sum(1 * forest == 3, axis=(1, 2)) / npoints
    per_dead = 100.0 * np.sum(1 * forest == 0, axis=(1, 2)) / npoints

    return per_bare, per_tree, per_burn, per_dead


def plot_forest(forest, istep=0, title='Forest Status', figsize=(7, 6)):
    '''
    Given a forest object, plot a 2D representation of the fire.

    Parameters
    ==========
    forest : Numpy array
        A "forest" array as generated from the forest_fire
    '''

    # Create a reasonably-sized figure:
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Plot the color map:
    map = ax.pcolor(forest[istep, :, :], vmin=1, vmax=3, cmap=forest_cmap)

    # Add a colorbar:
    cbar = plt.colorbar(map, ax=ax, shrink=.8, fraction=.08,
                        location='bottom', orientation='horizontal')
    cbar.set_ticks([1, 2, 3])
    cbar.set_ticklabels(['Bare/Burnt', 'Forested', 'Burning'])

    # Configure axes/labels:
    ax.set_title(title + f' (iStep={istep})')  # add iteration to title
    ax.set_aspect('equal')  # Set uniform aspect ratio.
    ax.set_xlabel('X (km)')  # Label X/Y assuming km.
    ax.set_ylabel('Y (km)')

    # Fig things snuggly into figure:
    fig.tight_layout()

    return fig, ax


def plot_disease(forest, istep=0, title='Disease Status', figsize=(7, 6)):
    '''
    Given a forest object, plot a 2D representation of disease spread.
    '''

    # Create a reasonably-sized figure:
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Plot the color map:
    map = ax.pcolor(forest[istep, :, :], vmin=0, vmax=3, cmap=forest_sick)

    # Add a colorbar:
    cbar = plt.colorbar(map, ax=ax, shrink=.8, fraction=.08,
                        location='bottom', orientation='horizontal')
    cbar.set_ticks([0, 1, 2, 3])
    cbar.set_ticklabels(['Dead', 'Immune', 'Healthy', 'Infected'])

    # Configure axes/labels:
    ax.set_title(title + f' (iStep={istep})')  # add iteration to title
    ax.set_aspect('equal')  # Set uniform aspect ratio.

    # Fig things snuggly into figure:
    fig.tight_layout()

    return fig, ax


def plot_forest_index(forest, istep=0, title='Forest Status', figsize=(7, 6)):
    '''
    Given a forest object, plot a 2D representation of the fire with cell
    indexes and fire statuses.
    THIS IS FOR DEMO PURPOSES WHILE WRITING THE LAB.
    '''

    # Get size of forest array:
    ny, nx = forest.shape[1:]

    # Create a reasonably-sized figure:
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Plot the color map:
    ax.pcolor(forest[istep, :, :], vmin=1, vmax=3, cmap=forest_cmap)

    # Now, label each cell:
    for i in range(nx):
        for j in range(ny):
            # Print indices onto cell:
            ax.text(i+.5, j+.35, f"i, j = {i:d}, {j:d}", size=10, ha='center')
            # Print status onto cell:
            s = status[forest[istep, j, i]]
            ax.text(i+.5, j+.6, s, size=14, ha='center')

    # Configure axes/labels:
    ax.set_title(title + f' (iStep={istep})')  # add iteration to title
    ax.set_aspect('equal')  # Set uniform aspect ratio.
    ax.set_xlabel('X (km)')  # Label X/Y assuming km.
    ax.set_ylabel('Y (km)')

    # Fig things snuggly into figure:
    fig.tight_layout()

    return fig, ax


def plot_percents(forest, title=''):
    '''
    Given a forest array, calculate and plot the percent of the forest that has
    trees, is bare, or is burning as a function of iteration.

    Parameters
    ==========
    forest : Numpy array
        A 3D array of forest status as generated by forest_fire

    Other Parameters
    ================
    title : string
        Set the title of the plot. Defaults to empty string.

    Returns
    =======
    fig : Matplotlib figure object
    ax : Matplotlib axes object
    '''

    # Get percentages:
    per_bare, per_tree, per_burn = calc_percents(forest)

    # Create array of iterations:
    iters = np.arange(forest.shape[0])

    # Create figure and axes:
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    # Plot each percentage vs. iteration:
    ax.plot(iters, per_bare, label='Bare/Burnt')
    ax.plot(iters, per_tree, label='Forested')
    ax.plot(iters, per_burn, label='Burning')

    # Add labels, legends, etc.
    ax.legend(loc='best')
    ax.set_xlabel('Time (iterations)')
    ax.set_ylabel('Percent of Total Area')
    ax.set_title(title)

    # Tight layout makes things look cleaner:
    fig.tight_layout()

    return fig, ax


def plot_percents_disease(forest, title=''):
    '''
    Given a forest array used to simulate a disease, plot the percent of
    people that are sick, dead, immune, or healthy as a function of iteration.

    Parameters
    ==========
    forest : Numpy array
        A 3D array of disease status as generated by forest_fire

    Other Parameters
    ================
    title : string
        Set the title of the plot. Defaults to empty string.

    Returns
    =======
    fig : Matplotlib figure object
    ax : Matplotlib axes object
    '''

    # Get percentages:
    per_imm, per_hlt, per_sck, per_ded = calc_percents_disease(forest)

    # Create array of iterations:
    iters = np.arange(forest.shape[0])

    # Create figure and axes:
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    # Plot each percentage vs. iteration:
    ax.plot(iters, per_imm, c=colors_sick[1], label='Immune')
    ax.plot(iters, per_hlt, c=colors_sick[2], label='Healthy')
    ax.plot(iters, per_sck, c=colors_sick[3], label='Sick')
    ax.plot(iters, per_ded, c=colors_sick[0], label='Dead')

    # Add labels, legends, etc.
    ax.legend(loc='best')
    ax.set_xlabel('Time (iterations)')
    ax.set_ylabel('Percent of Total Population')
    ax.set_title(title)

    # Tight layout makes things look cleaner:
    fig.tight_layout()

    return fig, ax


def test_forest(outdir=None):
    '''
    A unit test to ensure forest is working as expected.

    Parameters
    ==========
    None

    Other Parameters
    ================
    outdir : Location on disk
        If present, all figures will be saved to disk at `outdir`.

    Returns
    =======
    None

    '''

    # Test 1: 3x3 grid/100% burn/center ignition
    forest1 = forest_fire(ignite=[[1, 1]])

    # Generate forest plots:
    for i in range(3):
        fig, ax = plot_forest(forest1, title='3x3 Test', istep=i)
        if outdir:
            fig.savefig(outdir + f'/forest_3x3test_iter{i:02d}.png')

    # Plot percentage dynamics:
    fig, ax = plot_percents(forest1, title='3x3 Test')
    if outdir:
        fig.savefig(outdir + '/percents_3x3test.png')

    # Test2: 5x3 test (wider than tall), same params as above:
    forest2 = forest_fire(nx=5, ignite=[[2, 1]], nstep=6)

    # Generate forest plots:
    for i in range(3):
        fig, ax = plot_forest(forest2, title='5x3 Test', istep=i)
        if outdir:
            fig.savefig(outdir + f'/forest_5x3test_iter{i:02d}.png')

    # Plot percentage dynamics:
    fig, ax = plot_percents(forest2, title='5x3 Test')
    if outdir:
        fig.savefig(outdir + '/percents_5x3test.png')

    # Return both forests to caller.
    return forest1, forest2


def forest_demo():
    '''
    Create a high-res demo that looks cool. Save figures in a folder.
    '''

    # Because I'm running a loop to make 100s of plots, I need to do some
    # tricks for handling memory. I'm going to switch to the non-interactive
    # "Agg" backend, but first record what backend I was using before and
    # restore that before quitting.
    import matplotlib
    backend = matplotlib.backends.backend
    matplotlib.use('Agg')

    # Set parameters:
    nx, ny = 1024, 512
    nstep = 500
    outdir = "./demo_frames/"

    # os is a module for working with the operating system.
    # Here, we see if the output directory exists. If not. create it.
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    # Because we're making so many plots, we want to turn off interactive
    # plotting. Save the interactive mode status here so we can turn it back
    # on when we're done (that's polite.)
    inter_status = plt.isinteractive()
    plt.ioff()

    # Create a very large forest fire:
    forest = forest_fire(nx=nx, ny=ny, nstep=nstep, prob_bare=0.125,
                         prob_spread=1.0, prob_start=0.00003)
    print(f"Simulation finished in {forest.shape[0]} iterations.")

    # Plot percent of forest that is burning/intact/bare.
    # Save this figure to file:
    plot_percents(forest)
    fig_per = plt.gcf()
    fig_per.savefig(outdir + 'percents.png')

    # Create many figures and save:
    for i in range(forest.shape[0]):
        # Print status to screen:
        print(f"Saving figure {i:03d}/{nstep:03d}")

        # Create a plot for the current step with custom size, save to disk.
        fig, ax = plot_forest(forest, istep=i, figsize=(14.4, 8.7))
        fig.savefig(outdir + f'frame_{i:04d}.png')

        # When making many many plots, it's good to close them when you're
        # done- too many at once will cause Matplotlib to crash.
        plt.close('all')

    # If we were in Pyplot interactive mode previously, return to that state.
    if inter_status:
        plt.ion()

    # Restore previous backend:
    matplotlib.use(backend)

    # Return the big 3D array to the caller.
    return forest


def test_disease(outdir=None):
    '''
    A unit test to ensure forest is working as expected.

    Parameters
    ==========
    None

    Other Parameters
    ================
    outdir : Location on disk
        If present, all figures will be saved to disk at `outdir`.

    Returns
    =======
    None

    '''

    # Test 1: 3x3 grid/100% burn/center ignition
    forest1 = forest_fire(ignite=[[1, 1]], prob_fatal=.5)

    # Generate forest plots:
    for i in range(3):
        fig, ax = plot_disease(forest1, title='3x3 Test', istep=i)
        if outdir:
            fig.savefig(outdir + f'/disease_3x3test_iter{i:02d}.png')

    # Plot percentage dynamics:
    fig, ax = plot_percents_disease(forest1, title='3x3 Test')
    if outdir:
        fig.savefig(outdir + '/percents_3x3test_disease.png')

    # Test2: 5x3 test (wider than tall), same params as above:
    forest2 = forest_fire(nx=5, ignite=[[2, 1]], nstep=6, prob_fatal=.5)

    # Generate forest plots:
    for i in range(3):
        fig, ax = plot_disease(forest2, title='5x3 Test', istep=i)
        if outdir:
            fig.savefig(outdir + f'/disease_5x3test_iter{i:02d}.png')

    # Plot percentage dynamics:
    fig, ax = plot_percents_disease(forest2, title='5x3 Test')
    if outdir:
        fig.savefig(outdir + '/percents_5x3test_disease.png')

    # Return both forests to caller.
    return forest1, forest2


def disease_demo():
    '''
    Similar to above, but for diseases.
    '''

    # Import matplotlib, change backend.
    import matplotlib
    backend = matplotlib.backends.backend
    matplotlib.use('Agg')

    # Save backend, turn on non-interactive mode.
    inter_status = plt.isinteractive()
    plt.ioff()

    # Set parameters:
    nx, ny = 512, 256  # 1024, 512
    nstep = 500
    outdir = "./demo_frames_disease/"

    # Here, we see if the output directory exists. If not. create it.
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    # Create a very large disease with a lot of initial immunity:
    forest = forest_fire(nx=nx, ny=ny, nstep=nstep, prob_bare=0.60,
                         prob_spread=1.0, prob_fatal=.5, prob_start=0.0003)
    print(f"Simulation finished in {forest.shape[0]} iterations.")

    # Plot percent of forest that is burning/intact/bare.
    # Save this figure to file:
    plot_percents_disease(forest)
    fig_per = plt.gcf()
    fig_per.savefig(outdir + 'percents.png')

    # Create many figures and save:
    for i in range(forest.shape[0]):
        # Print status to screen:
        print(f"Saving figure {i:03d}/{nstep:03d}")

        # Create a plot for the current step with custom size, save to disk.
        fig, ax = plot_disease(forest, istep=i, figsize=(14.4, 8.7))
        fig.savefig(outdir + f'frame_{i:04d}.png')

        # When making many many plots, it's good to close them when you're
        # done- too many at once will cause Matplotlib to crash.
        plt.close('all')

    # If we were in Pyplot interactive mode previously, return to that state.
    if inter_status:
        plt.ion()

    # Restore previous backend:
    matplotlib.use(backend)

    # Return the big 3D array to the caller.
    return forest


def question_2(outdir=None):
    '''
    Use the functions above to answer question 2 from Lab 01.
    Two tasks are performed:
    1) Examine the effect of p_spread on fire spread.
    2) Examine the impact of initial bare ground on  spread.
    '''

    # Create parameters to control our simulations:
    nx, ny = 256, 256  # grid size
    p_start = .001  # .01% initial infection
    n = 2000  # maximum steps
    p_spread = 1.0  # Spread rate for varying bare spots simulation.

    # Create ranges of values to explore:
    prob_spread = np.arange(0, 1, 0.05)
    prob_bare = np.arange(0, 1, 0.05)

    # Create empty lists to hold our resulting observables:
    safe_spread = []  # Amount of forest untouched.
    time_spread = []  # Time to end-of-fire.
    for p in prob_spread:
        # Light up the fire:
        forest = forest_fire(nx=nx, ny=ny, prob_spread=p,
                             prob_start=p_start, nstep=n)
        # Save number of iterations until fire is out.
        time_spread.append(forest.shape[0])
        # Save surviving forest percent:
        init_forest = np.sum(1 * forest[0, :, :] == 2)
        safe_spread.append(100.0*np.sum(1*forest[-1, :, :] == 2) / init_forest)

    # Now repeat for initial bare regions.
    # Create empty lists to hold our resulting observables:
    safe_bare = []
    time_bare = []
    for p in prob_bare:
        # Light up the fire:
        forest = forest_fire(nx=nx, ny=ny, prob_spread=p_spread,
                             prob_bare=p, prob_start=p_start, nstep=n)
        # Save number of iterations until disease has passed.
        time_bare.append(forest.shape[0])
        # Save surviving forest percent:
        init_forest = np.sum(1 * forest[0, :, :] == 2)
        safe_bare.append(100.0*np.sum(1*forest[-1, :, :] == 2) / init_forest)

    # Plot these results on figure with two axes. We want "two" Y-axes for
    # each main axes so that we can plot two values on the same subplot even
    # though they have different units/ranges/values.
    fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(10, 7))
    # Double these axes using "twinx":
    ax2, ax4 = ax1.twinx(), ax3.twinx()

    # Plot forest survival rates and fire progression time vs spread rate:
    l1, = ax1.plot(100*prob_spread, safe_spread)
    l2, = ax2.plot(100*prob_spread, time_spread, c='crimson')
    # Set labels
    ax1.set_xlabel('Fire Spread Chance (%)')
    ax1.set_ylabel('Remaining Forest (%)', c=l1.get_color())
    ax2.set_ylabel('Fire Duration (iters)', c=l2.get_color())
    # Set Y-axes colors to match lines
    ax1.tick_params(axis='y', colors=l1.get_color())
    ax2.tick_params(axis='y', colors=l2.get_color())
    ax2.grid(False)

    # Plot survival rates and disease progression time vs. fatality rate.
    l3, = ax3.plot(100*prob_bare, safe_bare)
    l4, = ax4.plot(100*prob_bare, time_bare, c='crimson')
    # Set labels
    ax3.set_xlabel('Initial Bare Land (%)')
    ax3.set_ylabel('Remaining Forest (%)', c=l1.get_color())
    ax4.set_ylabel('Fire Duration (iters)', c=l2.get_color())
    # Set Y-axes colors to match lines
    ax3.tick_params(axis='y', colors=l3.get_color())
    ax4.tick_params(axis='y', colors=l4.get_color())
    ax4.grid(False)

    # Snappy title:
    ax1.set_title('Fire Evolution vs. Fire Spread Rate & Bare Land Area')
    fig.tight_layout()
    if outdir is not None:
        fig.savefig(outdir + 'question2_figure.png')

    return time_spread, safe_spread


def question_3(outdir=None):
    '''
    Use the functions above to answer question 3 from Lab 01.
    Two tasks are performed:
    1) Examine the effect of p_fatal on disease spread.
    2) Examine the impact of initial immunity on disease spread.
    '''

    # Create parameters to control our simulations:
    nx, ny = 256, 256
    p_spread = 1   # This is a really strong disease.
    p_start = .001  # .01% initial infection
    p_fatal = 1.
    n = 1000

    # Create ranges of values to explore:
    prob_fatals = np.arange(0, 1, 0.1)
    prob_vaxed = np.arange(0, 1, 0.05)

    # Create empty lists to hold our resulting observables:
    rates_fatal = []
    times_fatal = []
    for p in prob_fatals:
        # Light up the disease:
        population = forest_fire(nx=nx, ny=ny, prob_spread=p_spread,
                                 prob_fatal=p, prob_start=p_start, nstep=n)
        # Save number of iterations until disease has passed.
        times_fatal.append(population.shape[0])
        # Save survival rate:
        rates_fatal.append(100.0 * np.sum(
            1 * population[-1, :, :] != 0) / (nx * ny))

    # Now repeat for varying vaccination rates:
    # Create empty lists to hold our resulting observables:
    rates_vaxed = []
    times_vaxed = []
    for p in prob_vaxed:
        # Light up the disease:
        population = forest_fire(nx=nx, ny=ny, prob_spread=p_spread,
                                 prob_bare=p, prob_fatal=p_fatal,
                                 prob_start=p_start, nstep=n)
        # Save number of iterations until disease has passed.
        times_vaxed.append(population.shape[0])
        # Save survival rate:
        rates_vaxed.append(100.0 * np.sum(
            1 * population[-1, :, :] != 0) / (nx * ny))

    # Plot these results on figure with two axes. We want "two" Y-axes for
    # each main axes so that we can plot two values on the same subplot even
    # though they have different units/ranges/values.
    fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(10, 7))
    # Double these axes using "twinx":
    ax2, ax4 = ax1.twinx(), ax3.twinx()

    # Plot survival rates and disease progression time vs. fatality rate.
    l1, = ax1.plot(100*prob_fatals, rates_fatal)
    l2, = ax2.plot(100*prob_fatals, times_fatal, c='crimson')
    # Set labels
    ax1.set_xlabel('Mortality Rate (%)')
    ax1.set_ylabel('Survior Rate (%)', c=l1.get_color())
    ax2.set_ylabel('Outbreak Duration (iters)', c=l2.get_color())
    # Set Y-axes colors to match lines
    ax1.tick_params(axis='y', colors=l1.get_color())
    ax2.tick_params(axis='y', colors=l2.get_color())
    ax2.grid(False)

    # Plot survival rates and disease progression time vs. fatality rate.
    l3, = ax3.plot(100*prob_vaxed, rates_vaxed)
    l4, = ax4.plot(100*prob_vaxed, times_vaxed, c='crimson')
    # Set labels
    ax3.set_xlabel('Vaccine Adoption Rate (%)')
    ax3.set_ylabel('Survior Rate (%)', c=l1.get_color())
    ax4.set_ylabel('Outbreak Duration (iters)', c=l2.get_color())
    # Set Y-axes colors to match lines
    ax3.tick_params(axis='y', colors=l3.get_color())
    ax4.tick_params(axis='y', colors=l4.get_color())
    ax4.grid(False)

    # Snappy title:
    ax1.set_title('Disease Characteristics vs. Mortality and Vaccine Rates')
    fig.tight_layout()
    if outdir is not None:
        fig.savefig(outdir + 'question3_figure.png')

    return times_vaxed, rates_vaxed


def create_all_report_results():
    '''
    This function executes all commands to produce all figures and output
    required to complete the lab. It places figures into a new folder called
    "lab_output/".
    '''

    # Because we're making so many plots, we want to turn off interactive
    # plotting. Save the interactive mode status here so we can turn it back
    # on when we're done (that's polite.)
    inter_status = plt.isinteractive()
    plt.ioff()

    outdir = 'report_results/'
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    # Generate figures for part 1 (testing):
    forest1, forest2 = test_forest(outdir=outdir)
    plt.close('all')

    # Generate some illustrative figures for the lab hand-out.
    f1, ax = plot_forest_index(forest1, istep=0)
    f2, ax = plot_forest_index(forest1, istep=1)
    f3, ax = plot_forest_index(forest1, istep=2)
    # Save those figures, too:
    for i, f in enumerate([f1, f2, f3]):
        f.savefig(outdir + f"forest_example_{i}.png")
    plt.close('all')

    # Create figures to answer questions:
    question_2(outdir=outdir)
    question_3(outdir=outdir)

    if inter_status:
        plt.ion()
