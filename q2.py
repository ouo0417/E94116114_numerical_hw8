import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

def f(x):
    return 0.5 * np.cos(x) + 0.25 * np.sin(2 * x)
    
phi = [
    lambda x: 1,
    lambda x: x,
    lambda x: x**2
]

a, b = -1, 1
G = np.zeros((3, 3))
for i in range(3):
    for j in range(3):
        G[i, j], _ = quad(lambda x: phi[i](x) * phi[j](x), a, b)
for i in range(3):
    F[i], _ = quad(lambda x: f(x) * phi[i](x), a, b)

coeffs = np.linalg.solve(G, F)
a0, a1, a2 = coeffs

def P2(x):
    return a0 + a1 * x + a2 * x**2

error, _ = quad(lambda x: (f(x) - P2(x))**2, a, b)
print("Final result:")
print(f"P2(x) = {a0:.6f} + {a1:.6f}x + {a2:.6f}x^2")
print(f"Least square error: {error:.6e}")

