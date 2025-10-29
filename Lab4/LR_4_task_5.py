import numpy as np 
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def generate_data(m):
    np.random.seed(42)
    X = 6 * np.random.rand(m, 1) - 4 
    y = 0.5 * X ** 2 + X + 2 + np.random.randn(m, 1)
    return X, y

m = 100
X, y = generate_data(m)

lin_reg = LinearRegression()
lin_reg.fit(X, y)
y_pred_lin = lin_reg.predict(X)

print("Лінійна регресія:")
print("intercept =", lin_reg.intercept_)
print("coef =", lin_reg.coef_)

print("MAE:", mean_absolute_error(y, y_pred_lin))
print("MSE:", mean_squared_error(y, y_pred_lin))
print("R2:", r2_score(y, y_pred_lin))

poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)

poly_reg = LinearRegression()
poly_reg.fit(X_poly, y)
y_pred_poly = poly_reg.predict(X_poly)

print("\nПоліноміальна регресія 2-го ступеня:")
print("intercept =", poly_reg.intercept_)
print("coef =", poly_reg.coef_)

print("MAE:", mean_absolute_error(y, y_pred_poly))
print("MSE:", mean_squared_error(y, y_pred_poly))
print("R2:", r2_score(y, y_pred_poly))

X_sorted_idx = X[:, 0].argsort()
X_sorted = X[X_sorted_idx]
y_pred_lin_sorted = y_pred_lin[X_sorted_idx]
y_pred_poly_sorted = y_pred_poly[X_sorted_idx]

plt.scatter(X, y, color='green', label='Дані')
plt.plot(X_sorted, y_pred_lin_sorted, color='blue', linewidth=2, label='Лінійна регресія')
plt.plot(X_sorted, y_pred_poly_sorted, color='red', linewidth=2, label='Поліноміальна регресія 2-го ступеня')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.title('Лінійна та поліноміальна регресії')
plt.show()
