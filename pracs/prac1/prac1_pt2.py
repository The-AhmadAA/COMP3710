import torch
import numpy as np
import matplotlib.pyplot as plt
import sys

iterations = 200

def main(fractal, zoomed=False):

    #device config 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # use numpy to create a 2D array of complex numbers on [-2,2]x[-2,2]
    if not zoomed:
        Y, X = np.mgrid[-1.3:1.3:0.005, -2:1:0.005] # default values according to the prac sheet
    # Y, X = np.mgrid[-3.14:3.14:0.005, -3.14:3.14:0.005] # for the sin version of a Newton Fractal
    else:
        Y, X = np.mgrid[0.5:1:0.00025, -0.8:0.2:0.00025] # zoomed in version

    # load into pytorch tensors
    x = torch.Tensor(X)
    y = torch.Tensor(Y)
    z = torch.complex(x, y) # important!
    zs = z.clone() # Updated!
    ns = torch.zeros_like(z)

    # transfer to the device
    z = z.to(device)
    zs = zs.to(device)
    ns = ns.to(device)

    # Mandelbrot set
    for i in range(iterations):
        # compute the new values of z: z^2 + x
        if fractal == "mandelbrot":
            zs_ = zs * zs + z
        elif fractal == "julia":
            zs_ = zs * zs - 0.8 + 0.156j
        
        # Have we diverged with this new value?
        not_diverged = torch.abs(zs_) < 4.0

        # Update variables to compute
        ns += not_diverged
        zs = zs_

    # plot
    plt.figure(figsize=(16, 10))

    plt.imshow(processFractal(ns.cpu().numpy()))
    plt.tight_layout(pad=0)
    plt.show()


def processFractal(a):
    """Display an array of iteration counts as a colourflul picture of a fractal"""
    a_cyclic = (6.28 * a / 20.0).reshape(list(a.shape) + [1])
    img = np.concatenate([10 + 20*np.cos(a_cyclic), 30 + 50 * np.sin(a_cyclic), 155 - 80 * np.cos(a_cyclic)], 2)
    img[a==a.max()] = 0
    a = img
    a = np.uint8(np.clip(a, 0, 255))
    return a


if __name__ == "__main__":
    fractals = ["mandelbrot", "julia"]
    print(sys.argv)
    
    if sys.argv[1] in fractals:
        main(sys.argv[1], True if len(sys.argv) == 3 and sys.argv[2].lower() == "true" else False)
    else:
        print("Invalid or No fractal specified\nusage: python prac1_pt2.py [fractal name]\nchoose from 'mandelbrot' or 'julia'")