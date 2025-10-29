import numpy as np 
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

def plot_learning_curves (model, X, y): 
    X_train, X_val, y_train, y_val = train_test_split (X, y, test_size=0.2, random_state=42)
    train_errors, val_errors = [], []

    for m in range (1, len (X_train)):
        model.fit (X_train [:m], y_train [:m])
        y_train_predict = model.predict (X_train [:m]) 
        y_val_predict = model.predict (X_val)
        train_errors.append(mean_squared_error (y_train_predict, y_train [:m]))
        val_errors.append (mean_squared_error (y_val_predict, y_val)) 
    
    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="Навчальний набір (помилка)") 
    plt.plot (np.sqrt(val_errors), "b-", linewidth=3, label="Перевірочний набір (помилка)")
    plt.legend()
    plt.xlabel("Кількість навчальних даних")
    plt.ylabel("RMSE (середньоквадратична помилка)")
    plt.grid(True)
    plt.show()
    
def generate_data(m):
    np.random.seed(42)
    X = 6 * np.random.rand(m, 1) - 4 
    y = 0.5 * X ** 2 + X + 2 + np.random.randn(m, 1)
    return X, y

m = 100
X, y = generate_data(m)

lin_reg = LinearRegression() 
plot_learning_curves (lin_reg, X, y)

polynomial_regression_10 = Pipeline([
    ("poly_features", PolynomialFeatures(degree=10, include_bias=False)),
    ("lin_reg", LinearRegression()),
])
plot_learning_curves(polynomial_regression_10, X, y)

polynomial_regression_2 = Pipeline([
    ("poly_features", PolynomialFeatures(degree=2, include_bias=False)),
    ("lin_reg", LinearRegression()),
])
plot_learning_curves(polynomial_regression_2, X, y)