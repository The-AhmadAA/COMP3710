"""
Newton Fractal Python code from https://scipython.com/book2/chapter-8-scipy/examples/the-newton-fractal/
Adapted by Ahmad Abu-Aysha to use pytorch for parallelization
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import torch

# A list of colors to distinguish the roots.
colors = ['b', 'r', 'g', 'y']

# the threshold delta for convergence to a root
TOL = 1.e-8

def newton(z0, f, fprime, MAX_IT=100):
    """
    The Newton-Raphson method applied to f(z).

    Returns the root found, starting with an initial guess, z0, or False
    if no convergence to tolerance TOL was reached within MAX_IT iterations.

    """

    # z = z0
    # for _ in range(MAX_IT):
    #     dz = f(z)/fprime(z)
    #     if abs(dz) < TOL:
    #         return z
    #     z -= dz
    # return False

    # modify this function to work with a tensor rather than a fixed value
    dzs = torch.ones_like(z0)
    for _ in range(MAX_IT):
        # compute the next guess
        dzs = f(z0)/fprime(z0) 

        # check if the change is below the threshold
        dzs = torch.where(torch.abs(dzs) < TOL, dzs, 0)

        z0 -= dzs
    return z0
        # for every subsequent operation, apply the iteration only if the abs(dzs) value is > TOL
    



def plot_newton_fractal(f, fprime, n=400, domain=(-1, 1, -1, 1)):
    """Plot a Newton Fractal by finding the roots of f(z).

    The domain used for the fractal image is the region of the complex plane
    (xmin, xmax, ymin, ymax) where z = x + iy, discretized into n values along
    each axis.

    """
    device = torch.device('gpu' if torch.cuda.is_available() else 'cpu')
    xmin, xmax, ymin, ymax = domain
    Y, X = np.mgrid[xmin:xmax:1/n, ymin:ymax:1/n]
    

    roots = []
    m = np.zeros((n, n))

    def get_root_index(roots, r):
        """
        Get the index of r in the list roots.

        If r is not in roots, append it to the list.
        """
        try:
            return np.where(np.isclose(roots, r, atol=TOL))[0][0]
        except IndexError:
            roots.append(r)
            return len(roots) - 1

    # set up the complex plane of values
    x = torch.Tensor(X)
    y = torch.Tensor(Y)
    z = torch.complex(x, y)
    # zs = z.clone()

    # set up tensors for the roots(?) and for the root indices
    rs = torch.zeros_like(z)
    ms = torch.zeros_like(z)
    

    # transfer to the device
    z = z.to(device)
    # zs = zs.to(device)
    rs = rs.to(device)
    ms = ms.to(device)

    # perform newton's method on all zs
    rs = newton(z, f, fprime)

    # ms = get_root_index(roots, rs)

    # convert this to parallelise the operation

    # for ix, x in enumerate(np.linspace(xmin, xmax, n)):
    #     for iy, y in enumerate(np.linspace(ymin, ymax, n)):
    #         z0 = x + y*1j
    #         r = newton(z0, f, fprime)
    #         if r is not False:
    #             ir = get_root_index(roots, r)
    #             m[iy, ix] = ir


    nroots = len(roots)



    if nroots > len(colors):
        # Use a "continuous" colormap if there are too many roots.
        cmap = 'hsv'
    else:
        # Use a list of colors for the colormap: one for each root.
        cmap = ListedColormap(colors[:nroots])
    # plt.imshow(ms, cmap=cmap, origin='lower')
    # plt.axis('off')
    # plt.show()


f = lambda z: z**2 + 3
fprime = lambda z: 2*z

plot_newton_fractal(f, fprime, n=800, domain=(-1, 1, -1, 1))
