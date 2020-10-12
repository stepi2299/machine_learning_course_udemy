import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor


dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
Y = dataset.iloc[:, -1].values

# Training the Random Forest Regression model on the whole dataset (only here we have difference from decision tree)
regressor = RandomForestRegressor(n_estimators=10, random_state=0)  # the most important is number of trees (N_estimators)
regressor.fit(X, Y)

# Predict the new result
y_pred = regressor.predict([[6.5]])
print(y_pred)

# Visualising the Decision Tree Regression results (high resolution)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, Y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
