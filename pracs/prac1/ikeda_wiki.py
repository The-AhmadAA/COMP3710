"""
Visualisation of the Ikeda Map Attractor, for Fractal assessment of COMP3710
Based on the Wikipedia algorithm for the Ikeda Map at

https://en.wikipedia.org/wiki/Ikeda_map

converted to run in PyTorch

Author: Ahmad Abu-Aysha

"""

import math
import torch
import matplotlib.pyplot as plt
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(u: float, points=200, iterations=1000, nlim=40, limit=False, title=True):
    """
    Args:
        u:float
            ikeda parameter
        points:int
            number of starting points
        iterations:int
            number of iterations
        nlim:int
            plot these many last points for 'limit' option. Will plot all points if set to zero
        limit:bool
            plot the last few iterations of random starting points if True. Else Plot trajectories.
        title:[str, NoneType]
            display the name of the plot if the value is affirmative
    """
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    X, Y = np.mgrid[-5:5:0.05, -5:5:0.05]
    x = torch.Tensor(X)
    y = torch.Tensor(Y)

    # X = 10 * np.random.randn(points, 1)
    # Y = 10 * np.random.randn(points, 1)

    x = x.to(device)
    y = y.to(device)
    
    # for n in range(points):
        # X = compute_ikeda_trajectory(u, x[n][0], y[n][0], iterations)
    K = compute_ikeda_trajectory(u, x, y, iterations)
    
    if limit:
        plot_limit(K, nlim)
        # tx, ty = 2.5, -1.8
    else:
        plot_ikeda_trajectory(K)
        # tx, ty = -30, -26
    
    plt.title(f"Ikeda Map ({u=:.2g}, {iterations=})") if title else None
    return plt

def compute_ikeda_trajectory(u, x, y, N):
    """Calculate a full trajectory

    Args:
        u - is the ikeda parameter
        x, y - coordinates of the starting point
        N - the number of iterations

    Returns:
        An array.
    """
    K = torch.Tensor()
    # xs = torch.Tensor(X)
    


    for n in range(N):
        # K[n] = np.array((x, y))
        
        t = 0.4 - 6 / (1 + x ** 2 + y ** 2)
        x1 = 1 + u * (x * torch.cos(t) - y * torch.sin(t))
        y1 = u * (x * torch.sin(t) + y * torch.cos(t))
        
        x = x1
        y = y1   
        # K[n] = (x, y)
        
    return x, y

def plot_limit(X, n: int) -> None:
    """
    Plot the last n points of the curve - to see end point or limit cycle

    Args:
        X: np.array
            trajectory of an associated starting-point
        n: int
            number of "last" points to plot
    """
    plt.plot(X[-n:, 0], X[-n:, 1], 'ko')

def plot_ikeda_trajectory(X) -> None:
    """
    Plot the whole trajectory

    Args:
        X: np.array
            trajectory of an associated starting-point
    """
    plt.plot(X[:,0], X[:, 1], "k")

if __name__ == "__main__":
    main(0.6, limit=False, nlim=0).show()