import torch
import numpy as np
import matplotlib.pyplot as plt

print("PyTorch Version: ", torch.__version__)

# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# grid for computing image, subdivide the space
X, Y = np.mgrid[-4.0: 4: 0.01, -4.0: 4: 0.01]

# load into PyTorch Tensors
x = torch.Tensor(X)
y = torch.Tensor(Y)

# transfer to the CPU/GPU
x = x.to(device)
y = y.to(device)


# Compute the Gaussian
# z = torch.exp(-(x**2 + y**2)/2.0)

# Compute the Sin
# z = torch.sin(x)

# Combine the Gaussian and a sin function?
z = torch.sin(x + y) * torch.exp(-(x**2 + y**2)/2.0)

#plot
plt.imshow(z.cpu().numpy())

plt.tight_layout()
plt.show()