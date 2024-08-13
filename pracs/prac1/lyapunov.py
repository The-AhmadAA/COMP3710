"""
    1. Choose a string of As and Bs of any nontrivial length (e.g., AABAB).
    2. Construct the sequence S {\displaystyle S} formed by successive terms in the string, repeated as many times as necessary.
    3. Choose a point ( a , b ) ∈ [ 0 , 4 ] × [ 0 , 4 ] {\displaystyle (a,b)\in [0,4]\times [0,4]}.
    4. Define the function r n = a {\displaystyle r_{n}=a} if S n = A {\displaystyle S_{n}=A}, and r n = b {\displaystyle r_{n}=b} if S n = B {\displaystyle S_{n}=B}.
    5. Let x 0 = 0.5 {\displaystyle x_{0}=0.5}, and compute the iterates x n + 1 = r n x n ( 1 − x n ) {\displaystyle x_{n+1}=r_{n}x_{n}(1-x_{n})}.
    6. Compute the Lyapunov exponent:
        λ = lim N → ∞ 1 N ∑ n = 1 N log ⁡ | d x n + 1 d x n | = lim N → ∞ 1 N ∑ n = 1 N log ⁡ | r n ( 1 − 2 x n ) | {\displaystyle \lambda =\lim _{N\rightarrow \infty }{1 \over N}\sum _{n=1}^{N}\log \left|{dx_{n+1} \over dx_{n}}\right|=\lim _{N\rightarrow \infty }{1 \over N}\sum _{n=1}^{N}\log |r_{n}(1-2x_{n})|}
        In practice, λ {\displaystyle \lambda } is approximated by choosing a suitably large N {\displaystyle N} and dropping the first summand as r 0 ( 1 − 2 x 0 ) = r n ⋅ 0 = 0 {\displaystyle r_{0}(1-2x_{0})=r_{n}\cdot 0=0} for x 0 = 0.5 {\displaystyle x_{0}=0.5}.
    7. Color the point ( a , b ) {\displaystyle (a,b)} according to the value of λ {\displaystyle \lambda } obtained.
    8. Repeat steps (3–7) for each point in the image plane.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt


def main(seq: str):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    X, Y = np.mgrid[0:4:0.005, 0:4:0.005]
    x = 
    x0 = 0.5




if __name__ == "__main__":

    main('AABAB')
    