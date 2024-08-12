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

def newton(z0, f, fprime, MAX_IT=100):
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
    Y, X = np.mgrid[xmin:xmax:1/n, ymin:ymax:1/n]

    def get_root_index(roots, r):
        """
        Get the index of r in the Tensor roots.

        If r is not in roots, concat the new root with the roots Tensor.
        """

        # roots_tensor = torch.Tensor(roots).clone()
        # TODO will need to modify this function so that the indices are returned?
        return torch.where(torch.isclose(roots, r, atol=TOL), roots, r)

    # set up the complex plane of values
    x = torch.Tensor(X)
    y = torch.Tensor(Y)
    z = torch.complex(x, y)
    roots = torch.empty((0,), dtype=torch.complex64, device=device)
    # set up tensors for the roots(?) and for the root indices
    rs = torch.zeros_like(z)
    ms = torch.zeros_like(z)
    

    # transfer to the device
    z = z.to(device)
    rs = rs.to(device)
    ms = ms.to(device)
    roots = roots.to(device)
    # perform newton's method on all zs
    rs = newton(z, f, fprime)
    # print(rs)
    # ===================================================================================
    # Tensor.unique doesn't work for complex values. Workaround involves separating into
    # real and imag parts, courtesy of chatGPT

    roots_real = rs.real
    roots_imag = rs.imag
    recombined = torch.stack((roots_real, roots_imag), dim=1)

    unique_roots = torch.unique(recombined, dim=0)

    unique_complex_roots = unique_roots[:, 0] + 1j * unique_roots[:, 1]
    print(unique_complex_roots)
    # ===============================================================================
    # allocate all the points to particular roots
    # ms = get_root_index(roots, rs)

    nroots = len(roots)



    if nroots > len(colors):
        # Use a "continuous" colormap if there are too many roots.
        cmap = 'hsv'
    else:
        # Use a list of colors for the colormap: one for each root.
        cmap = ListedColormap(colors[:nroots])
    # print(ms)
    # plt.imshow(ms, cmap=cmap, origin='lower')
    # plt.axis('off')
    # plt.show()


f = lambda z: z**3 + 3
fprime = lambda z: 3*z**2

plot_newton_fractal(f, fprime, n=800, domain=(-1, 1, -1, 1))
