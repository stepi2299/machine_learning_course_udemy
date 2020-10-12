import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values # we are taking all columns except last one
Y = dataset.iloc[:, -1].values  # values of the last column

# preparing dataset for feature scaling
print(Y)
Y = Y.reshape(len(Y), 1)  # changing feature format from horizontal to vertical
print(Y)

# Feature scaling
sc_X = StandardScaler()
sc_Y = StandardScaler()  # we are applying feature scaling also for dependent variable because earlier our values were binary
X = sc_X.fit_transform(X)
Y = sc_Y.fit_transform(Y)
print(X)
print(Y)

# Training the SVR model on the whole dataset
regressor = SVR(kernel = 'rbf')  # parameter is kernel
regressor.fit(X, Y)  # training

# Predict a new result
sc_Y.inverse_transform(regressor.predict(sc_X.transform([[6.5]])))  # reversing the feature scaling to give a result

# Visualising the SVR result
plt.scatter(sc_X.inverse_transform(X), sc_Y.inverse_transform(Y), color='red')
plt.plot(sc_X.inverse_transform(X), sc_Y.inverse_transform(regressor.predict(X)), color='blue')  # we have to change argument of predict function on a matrix of NOT single feature
plt.title('Truth or bluff (Support Vector Regression Model')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Visualising the SVR results (for higher resolution and smoother curve)
X_grid = np.arange(min(sc_X.inverse_transform(X)), max(sc_X.inverse_transform(X)), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(sc_X.inverse_transform(X), sc_Y.inverse_transform(Y), color='red')
plt.plot(X_grid, sc_Y.inverse_transform(regressor.predict(sc_X.transform(X_grid))), color='blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
