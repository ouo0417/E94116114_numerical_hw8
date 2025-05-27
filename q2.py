import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

# 定義函數 f(x)
def f(x):
    return 0.5 * np.cos(x) + 0.25 * np.sin(2 * x)

# 定義基底 phi0=1, phi1=x, phi2=x^2
phi = [
    lambda x: 1,
    lambda x: x,
    lambda x: x**2
]

# 積分區間
a, b = -1, 1

# 構建 Gram 矩陣 (左邊)
G = np.zeros((3, 3))
for i in range(3):
    for j in range(3):
        G[i, j], _ = quad(lambda x: phi[i](x) * phi[j](x), a, b)

# 右邊向量 (內積 f 與每個基底)
F = np.zeros(3)
for i in range(3):
    F[i], _ = quad(lambda x: f(x) * phi[i](x), a, b)

# 解線性系統 G a = F
coeffs = np.linalg.solve(G, F)
a0, a1, a2 = coeffs

# 構建逼近函數 P2(x)
def P2(x):
    return a0 + a1 * x + a2 * x**2

# 計算平方誤差
error, _ = quad(lambda x: (f(x) - P2(x))**2, a, b)

# 顯示結果
print("Final result:")
print(f"P2(x) = {a0:.6f} + {a1:.6f}x + {a2:.6f}x^2")
print(f"Least square error: {error:.6e}")

