"""
Newton Fractal Python code from https://scipython.com/book2/chapter-8-scipy/examples/the-newton-fractal/
Adapted by Ahmad Abu-Aysha to use pytorch for parallelization
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import torch

# a couple of settings to ensure that uninitialized values (roots) are filled with
# NaN instead of random values
torch.use_deterministic_algorithms(True)
torch.utils.deterministic.fill_uninitialized_memory = True

# A list of colors to distinguish the roots.
colors = ['b', 'r', 'g', 'y']

# the threshold delta for convergence to a root
TOL = 1.e-8

root_indices = []

def newton(z0, f, fprime, MAX_IT=1000):
    """
    The Newton-Raphson method applied to f(z).

    Returns the root found, starting with an initial guess, z0, or False
    if no convergence to tolerance TOL was reached within MAX_IT iterations.

    """

    # modify this function to work with a tensor rather than a fixed value
    dzs = torch.ones_like(z0)
    for _ in range(MAX_IT):
        # compute the next guess
        dzs = f(z0)/fprime(z0) 
        z0 -= dzs
        # # check if the change is below the threshold
        if torch.max(torch.abs(dzs)) < TOL:
            break

    return z0
        # for every subsequent operation, apply the iteration only if the abs(dzs) value is > TOL

def plot_newton_fractal(f, fprime, n=400, domain=(-1, 1, -1, 1)):
    """Plot a Newton Fractal by finding the roots of f(z).

    The domain used for the fractal image is the region of the complex plane
    (xmin, xmax, ymin, ymax) where z = x + iy, discretized into n values along
    each axis.

    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    xmin, xmax, ymin, ymax = domain
    Y, X = np.mgrid[ymin:ymax:(ymax - ymin)/n, xmin:xmax:(xmax - xmin)/n]

    # set up the complex plane of values
    x = torch.Tensor(X)
    y = torch.Tensor(Y)
    z = torch.complex(x, y)

    # set up tensors for the roots(?) and for the root indices
    rs = torch.zeros_like(z)
    ms = torch.zeros_like(z)
    

    # transfer to the device
    z = z.to(device)
    rs = rs.to(device)
    ms = ms.to(device)

    # perform newton's method on all zs
    rs = newton(z, f, fprime)

    # Can't easily find unique values as Tensor.unique doesn't support complex values - 
    # Workaround courtesy of chatGPT 
    # ================================================================================

    # Separate real and imaginary parts
    real_parts = rs.real
    imag_parts = rs.imag

    combined = torch.stack((real_parts, imag_parts), dim=-1).reshape(-1, 2)

    # rounding to truncate floating point values to ensure that the correct number of unique values found
    n_digits = 3
    rounded = (combined * 10**n_digits).round() / (10**n_digits)

    # Find unique rows and the corresponding indices
    unique_roots, inverse_indices = torch.unique(rounded, dim=0, return_inverse=True)

    print(unique_roots)

    # Reshape the indices back to the grid shape
    ms = inverse_indices.view(n, n).cpu().numpy()

    # ================================================================================
    # Assign color mapping to indices
    nroots = len(unique_roots)

    if nroots > len(colors):
        # Use a "continuous" colormap if there are too many roots.
        cmap = 'hsv'
    else:
        # Use a list of colors for the colormap: one for each root.
        cmap = ListedColormap(colors[:nroots])
    # print(ms)
    plt.imshow(ms, cmap=cmap, origin='lower')
    plt.axis('off')
    plt.show()


f = lambda z: z**3 + 3
fprime = lambda z: 3*z**2

plot_newton_fractal(f, fprime, n=1600, domain=(-0.5, 0.5, -0.5, 0.5))
