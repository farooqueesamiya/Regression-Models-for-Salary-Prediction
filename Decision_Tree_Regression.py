# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load the dataset from a CSV file
dataset = pd.read_csv('Position_Salaries.csv')

# Extract features (X) and target (y) values from the dataset
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Build a Decision Tree Regressor
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)

# Train the Decision Tree Regressor on the dataset
regressor.fit(X, y)

# Predict the salary for a specific position level
predicted_salary = regressor.predict([[6.5]])
print("Predicted Salary:", predicted_salary)

# Create a smooth X grid for plotting
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))

# Scatter plot of the original data points
plt.scatter(X, y, color='red')

# Plot the Decision Tree Regressor's prediction line
plt.plot(X_grid, regressor.predict(X_grid), color='blue')

# Add labels and title to the plot
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')

# Show the plot
plt.show()
