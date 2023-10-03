#!/usr/bin/env python3

'''
A quick demonstration of forward, backward, and central differencing
and the performance of each vs. `h` (aka, \Delta x)
'''

import numpy as np
from matplotlib import pyplot as plt

plt.style.use('fivethirtyeight')


def diff1(fx, deltax=1.0):
    '''
    For a series of values representing f(x), where each value is separated
    by some deltax, perform the forward difference approximation to the
    derivative. Backward diff is used at the final point.

    Parameters
    ----------
    fx : numpy array
      A series of values representing f(x) taken at even spacing of deltax

    Other Parameters
    ----------------
    deltax : float
      The x-spacing of points in fx; defaults to 1.0
    '''

    # Create result array equal in size to input:
    dfdx = np.zeros(fx.size)

    # Calculate forward difference for all points except the last:
    dfdx[:-1] = (fx[1:] - fx[:-1]) / deltax

    # Calculate the backward difference for the last point:
    dfdx[-1] = (fx[-1] - fx[-2]) / deltax

    # Return to caller:
    return dfdx


def centdiff(fx, deltax=1.0):
    '''
    For a series of values representing f(x), where each value is separated
    by some deltax, perform the central difference approximation to the
    derivative. 2nd order forward/backward diff is used at edges.

    Parameters
    ----------
    fx : numpy array
      A series of values representing f(x) taken at even spacing of deltax

    Other Parameters
    ----------------
    deltax : float
      The x-spacing of points in fx; defaults to 1.0
    '''

    # Create result array equal in size to input:
    dfdx = np.zeros(fx.size)

    # Calculate central difference for all non-edge:
    dfdx[1:-1] = (fx[2:] - fx[:-2]) / (2*deltax)

    # Calculate the forward difference for the first point:
    dfdx[0] = (3*fx[0] + 4*fx[1] - fx[2]) / (2*deltax)

    # Calculate the backward difference for the last point:
    dfdx[-1] = (3*fx[-1] - 4*fx[-2] + fx[-3]) / (2*deltax)

    # Return to caller:
    return dfdx


# Demo 1: Show how our two methods compare against each other.
# Start with a sin function:
x = np.linspace(0, 4*np.pi, 31)
y = np.sin(x)

# Get spacing:
dx = x[1] - x[0]

# Our analytical derivative:
dy = np.cos(x)

# Our numerical derivatives:
dy1 = diff1(y, dx)
dy2 = centdiff(y, dx)

# And plot:
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.plot(x, y, c='gray', label=r'$f(x)=\sin(x)$')
ax.plot(x, dy, c='k', label=r'$\frac{df(x)}{dx}=\cos(x)$')
ax.plot(x, dy1, label='Forward Diff')
ax.plot(x, dy2, label='Cent. Diff')
# ax.set_title('Numerical Differentiation: Forward Diff')
ax.set_title('Numerical Differentiation: 1st vs. 2nd Order')
ax.set_xlabel('$x$', size=20)
ax.set_ylabel('$f(x)$', size=20)
ax.legend(loc='best')
fig.tight_layout()
fig.savefig('demo02_numdiff.png')

# Demo 2: Show convergence rates
# We'll calculate derivatives for many dx values.
# We'll store the error from the analytical and look at convergence.

all_dx = np.linspace(0.0001, 1, 100)
err1 = np.zeros(all_dx.size)
err2 = np.zeros(all_dx.size)

# Loop over many many many dx values
for i, dx in enumerate(all_dx):
    # For each dx, calculate series and derivatives
    x = np.arange(0, 4*np.pi, dx)
    y = np.sin(x)
    dy = np.cos(x)
    dy1 = diff1(y, dx)
    dy2 = centdiff(y, dx)

    # Save maximum error for each:
    err1[i] = (np.abs(dy - dy1)).max()
    err2[i] = (np.abs(dy - dy2)).max()

# Show what it looks like as a plot:
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.loglog(all_dx, err1, label='1st Order Diff')
ax.loglog(all_dx, err2, label='2nd Order Diff')
ax.set_xlabel(r'$\Delta x$')
ax.set_ylabel('Error')
ax.set_title('Convergence of Numerical Differentiation vs. $\\Delta x$')
ax.legend(loc='best')
fig.tight_layout()
fig.savefig('demo02_numdiff_converge.png')