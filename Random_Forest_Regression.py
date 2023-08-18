# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load the CSV file dataset into a DataFrame
dataset = pd.read_csv('Position_Salaries.csv')

# Extract features (X) and target (y)
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Import the RandomForestRegressor class from the ensemble module
from sklearn.ensemble import RandomForestRegressor

# Create a RandomForestRegressor instance with 10 estimators (trees)
# and a random_state for reproducibility
regressor = RandomForestRegressor(n_estimators=10, random_state=0)

# Fit the regressor on the training data (X, y)
regressor.fit(X, y)

# Predict the salary for a specific position level (6.5)
prediction = regressor.predict([[6.5]])
print(f"Predicted Salary for Position Level 6.5: {prediction[0]:.2f}")

# Create a grid of positions for smoother visualization
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))

# Scatter plot of actual data points
plt.scatter(X, y, color='red')

# Plot the predictions on the grid
plt.plot(X_grid, regressor.predict(X_grid), color='blue')

# Add title and labels to the plot
plt.title('Truth or Bluff (Random Forest Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')

# Display the plot
plt.show()
