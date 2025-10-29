import numpy as np
import matplotlib.pyplot as plt

X = np.array([16, 27, 38, 19, 100, 72])
Y = np.array([12, 35, 39, 41, 60, 55])
n = len(X)

# Обчислення коефіцієнтів за формулами методу найменших квадратів
beta1 = (n * np.sum(X*Y) - np.sum(X) * np.sum(Y)) / (n * np.sum(X**2) - (np.sum(X))**2)
beta0 = np.mean(Y) - beta1 * np.mean(X)

# Виведення результатів обчислень
print(f"β0 = {beta0:.2f}")
print(f"β1 = {beta1:.2f}")
print(f"Рівняння прямої: y = {beta0:.2f} + {beta1:.2f}x")

# Прогнозні значення
Y_pred = beta0 + beta1 * X
S = np.sum((Y - Y_pred)**2)
print(f"S({beta0:.2f}; {beta1:.2f}) = {S:.6f}")

# Візуалізація
plt.scatter(X, Y, color='blue', label='Експериментальні точки')
plt.plot(X, Y_pred, color='red', label='Апроксимація (МНК)')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Метод найменших квадратів')
plt.legend()
plt.grid(True)
plt.show()