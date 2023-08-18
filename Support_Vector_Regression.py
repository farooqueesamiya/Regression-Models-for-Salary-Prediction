# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load the CSV file dataset into a DataFrame
dataset = pd.read_csv('Position_Salaries.csv')

# Extract features (Position Level) and target (Salary)
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Reshape the target y to a 2D array
y = y.reshape(len(y), 1)

# Import StandardScaler class from preprocessing module
from sklearn.preprocessing import StandardScaler

# Create separate scalers for features (X) and target (y)
sc_X = StandardScaler()
sc_y = StandardScaler()

# Scale the features (X) and target (y)
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

# Import SVR class from svm module
from sklearn.svm import SVR

# Create an SVR instance with a radial basis function (RBF) kernel
regressor = SVR(kernel='rbf')

# Fit the SVR regressor on the scaled data
regressor.fit(X, y)

# Predict the scaled salary for a given position level and inverse transform to get the original scale
scaled_predicted_salary = regressor.predict(sc_X.transform([[6.5]])).reshape(-1, 1)
predicted_salary = sc_y.inverse_transform(scaled_predicted_salary)

# Scatter plot of original scale data points
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color='red')

# Plot the SVR predictions and inverse transform to get original scale predictions
plt.plot(sc_X.inverse_transform(X), sc_y.inverse_transform(regressor.predict(X).reshape(-1, 1)), color='blue')

# Add title and labels to the plot
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')

# Display the plot
plt.show()

# Create a grid of position levels for smoother visualization
X_grid = np.arange(min(sc_X.inverse_transform(X)), max(sc_X.inverse_transform(X)), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))

# Scatter plot of original scale data points
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color='red')

# Plot the SVR predictions on the grid and inverse transform to get original scale predictions
plt.plot(X_grid, sc_y.inverse_transform(regressor.predict(sc_X.transform(X_grid)).reshape(-1, 1)), color='blue')

# Add title and labels to the plot
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')

# Display the plot
plt.show()
