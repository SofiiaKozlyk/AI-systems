import numpy as np
import matplotlib.pyplot as plt

x = np.array([0.1, 0.3, 0.4, 0.6, 0.7])
y = np.array([3.2, 3, 1, 1.8, 1.9])

# Формування матриці Вандермонда
X = np.vander(x, increasing=True)
print(f"Матриця Вандермонда:\n{X}")

# Розв'язання системи для знаходження коефіцієнтів інтерполяційного полінома
coeffs = np.linalg.solve(X, y)
print(f"Коефіцієнти інтерполяційного полінома:\n{coeffs}")

# Визначення функції полінома
polynomial = np.poly1d(coeffs[::-1])
print(f"Функція інтерполяційного полінома:\n{polynomial}")

# Візуалізація
x_plot = np.linspace(min(x)-0.05, max(x)+0.05, 200)
y_plot = polynomial(x_plot)

plt.figure(figsize=(8,5))
plt.plot(x_plot, y_plot, label="Інтерполяційний поліном", color='blue')
plt.scatter(x, y, color='red', label="Експериментальні точки")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Інтерполяція функції поліномом 4-го степеня")
plt.legend()
plt.grid(True)
plt.show()

# Обчислення значень в проміжних точках 0.2 і 0.5
print("Значення функції в проміжних точках:")
x_test = [0.2, 0.5]
y_test = polynomial(x_test)
for xt, yt in zip(x_test, y_test):
    print(f"P({xt}) = {yt:.4f}")
