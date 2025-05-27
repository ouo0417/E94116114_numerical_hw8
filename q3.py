import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

# 原始函數 f(x) = x^2 * sin(x)
def f(x):
    return x**2 * np.sin(x)

# 離散點數設定：m = 16, 所以使用 2m = 32 個點
m = 16
N = 2 * m
x_j = np.linspace(0, 1, N, endpoint=False)
y_j = f(x_j)

# 建立三角多項式 S_4(x) 的係數：a_0, a_k, b_k
a_0 = (1 / m) * np.sum(y_j)

a_k = []
b_k = []

for k in range(1, 5):  # k = 1 to 4
    a_k_val = (1 / m) * np.sum(y_j * np.cos(2 * np.pi * k * x_j))
    b_k_val = (1 / m) * np.sum(y_j * np.sin(2 * np.pi * k * x_j))
    a_k.append(a_k_val)
    b_k.append(b_k_val)

# 定義 S_4(x)
def S4(x):
    result = 0.5 * a_0
    for k in range(1, 5):
        result += a_k[k - 1] * np.cos(2 * np.pi * k * x)
        result += b_k[k - 1] * np.sin(2 * np.pi * k * x)
    return result

# (b) 積分 ∫_0^1 S_4(x) dx
integral_S4, _ = quad(S4, 0, 1)

# (c) 積分 ∫_0^1 x^2 sin(x) dx
integral_exact, _ = quad(f, 0, 1)

# (d) 誤差 E(S4) = sum [f(x_j) - S_4(x_j)]^2
error = np.sum((y_j - S4(x_j))**2)

# 顯示結果
print("=== S_4(x) approximation results ===")
print(f"∫₀¹ S₄(x) dx ≈ {integral_S4:.6f}")
print(f"∫₀¹ x² sin(x) dx ≈ {integral_exact:.6f}")
print(f"Discrete least squares error E(S₄) = {error:.6f}")

