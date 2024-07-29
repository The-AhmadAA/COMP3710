import torch
import numpy as np
import matplotlib.pyplot as plt

#device config 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# use numpy to create a 2D array of complex numbers on [-2,2]x[-2,2]

Y, X = np.mgrid[-1.3:1.3:0.005, -2:1:0.005]

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
for i in range(200):
    # compute the new values of z: z^2 + x
    zs_ = zs * zs + z
    
    # Have we diverged with this new value?
    not_diverged = torch.abs(zs_) < 4.0

    # Update variables to compute
    ns += not_diverged
    zs = zs_

# plot
fig = plt.figure(figsize=(16, 10))

def processFractal(a):
    """Display an array of iteration countes as a colourflul picture of a fractal"""
    a_cyclic = (6.28 * a / 20.0).reshape(list(a.shape) + [1])
    img = np.concatenate([10 + 20*np.cos(a_cyclic), 30 + 50 * np.sin(a_cyclic), 155 - 80 * np.cos(a_cyclic)], 2)
    img[a==a.max()] = 0
    a = img
    a = np.uint8(np.clip(a, 0, 255))
    return a

plt.imshow(processFractal(ns.cpu().numpy()))
plt.tight_layout(pad=0)
plt.show()