import numpy as np
import matplotlib.pyplot as plt

# 原始資料
x = np.array([4.0, 4.2, 4.5, 4.7, 5.1, 5.5, 5.9, 6.3])
y = np.array([102.6, 113.2, 130.1, 142.1, 167.5, 195.1, 224.9, 256.8])

### (a) Degree 2 polynomial approximation
coeffs_quad = np.polyfit(x, y, 2)              # 二次多項式係數
p_quad = np.poly1d(coeffs_quad)                # 多項式函數
y_quad = p_quad(x)                             # 擬合結果
error_quad = np.sum((y - y_quad)**2)           # 平方誤差

### (b) Exponential approximation y ≈ b * e^(a*x)
ln_y = np.log(y)
A_exp = np.vstack([x, np.ones(len(x))]).T
a_exp, ln_b_exp = np.linalg.lstsq(A_exp, ln_y, rcond=None)[0]
b_exp = np.exp(ln_b_exp)
y_exp = b_exp * np.exp(a_exp * x)
error_exp = np.sum((y - y_exp)**2)

### (c) Power approximation y ≈ b * x^a
ln_x = np.log(x)
A_pow = np.vstack([ln_x, np.ones(len(ln_x))]).T
a_pow, ln_b_pow = np.linalg.lstsq(A_pow, ln_y, rcond=None)[0]
b_pow = np.exp(ln_b_pow)
y_pow = b_pow * x**a_pow
error_pow = np.sum((y - y_pow)**2)

# 印出結果
print("=== (a) Degree 2 Polynomial ===")
print(f"Model: y = {coeffs_quad[0]:.4f}x^2 + {coeffs_quad[1]:.4f}x + {coeffs_quad[2]:.4f}")
print(f"Squared Error: {error_quad:.4f}\n")

print("=== (b) Exponential Fit y = b * e^(a*x) ===")
print(f"Model: y = {b_exp:.4f} * e^({a_exp:.4f}x)")
print(f"Squared Error: {error_exp:.4f}\n")

print("=== (c) Power Fit y = b * x^a ===")
print(f"Model: y = {b_pow:.4f} * x^{a_pow:.4f}")
print(f"Squared Error: {error_pow:.4f}\n")

