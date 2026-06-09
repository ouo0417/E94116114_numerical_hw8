import numpy as np
from scipy.integrate import quad

def f(x):
    return x**2 * np.sin(x)


m = 16
N = 2 * m
x_j = np.linspace(0, 1, N, endpoint=False)
y_j = f(x_j)
a_0 = (1 / m) * np.sum(y_j)
a_k = []
b_k = []

for k in range(1, 4):
    a_k_val = (1 / m) * np.sum(y_j * np.cos(2 * np.pi * k * x_j))
    b_k_val = (1 / m) * np.sum(y_j * np.sin(2 * np.pi * k * x_j))
    a_k.append(a_k_val)
    b_k.append(b_k_val)

a_4 = (1 / m) * np.sum(y_j * np.cos(2 * np.pi * 4 * x_j))

def S4(x):
    result = 0.5 * a_0
    for k in range(1, 4):
        result += a_k[k - 1] * np.cos(2 * np.pi * k * x)
        result += b_k[k - 1] * np.sin(2 * np.pi * k * x)
    result += a_4 * np.cos(2 * np.pi * 4 * x)
    return result

integral_S4, _ = quad(S4, 0, 1)
integral_exact, _ = quad(f, 0, 1)
error = np.sum((y_j - S4(x_j))**2)

print()
print(f"Integral of S4 from 0 to 1 = {integral_S4:.10f}")
print(f"Integral of x^2 sin(x) from 0 to 1 = {integral_exact:.10f}")
print(f"Error E(S4) = {error:.10f}")
