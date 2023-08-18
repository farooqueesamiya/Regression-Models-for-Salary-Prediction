# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load the CSV file dataset into a DataFrame
dataset = pd.read_csv('Position_Salaries.csv')

# Extract the features (Position Level) and target (Salary)
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Perform Linear Regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Perform Polynomial Regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# Visualize Linear Regression results
plt.scatter(X, y, color='red')
plt.plot(X, lin_reg.predict(X), color='blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Visualize Polynomial Regression results
plt.scatter(X, y, color='red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color='blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualize Polynomial Regression results with a smoother curve
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color='blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Predict using both Linear and Polynomial Regression models
linear_prediction = lin_reg.predict([[6.5]])
polynomial_prediction = lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))

print(f"Linear Regression Prediction for Position Level 6.5: {linear_prediction[0]:.2f}")
print(f"Polynomial Regression Prediction for Position Level 6.5: {polynomial_prediction[0]:.2f}")
