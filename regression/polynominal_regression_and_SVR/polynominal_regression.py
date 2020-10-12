import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values  # we are taking all columns except last one
Y = dataset.iloc[:, -1].values  # values of the last column

lin_reg = LinearRegression()
lin_reg.fit(X, Y)  # here we are training our model

# Training the Polynominal Regression model on the whole dataset
poly_reg = PolynomialFeatures(degree=4)  # first x^2 but we can experiment (pow depends on degree)
X_poly = poly_reg.fit_transform(X)  # transforming matrix of a single feature into new matrix of features composed of x1 as first feature and x1^2 as second
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, Y)  # creation of polynominal regression model

# Visualising the Linear Regression result
plt.scatter(X, Y, color='red')
plt.plot(X, lin_reg.predict(X), color='blue')
plt.title('Truth or bluff (Linear Regression Model')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Visualising the Polynominal Regression result
plt.scatter(X, Y, color='red')
plt.plot(X, lin_reg2.predict(X_poly), color='blue')  # we have to change argument of predict function on a matrix of NOT single feature
plt.title('Truth or bluff (Polynominal Linear Regression Model')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Visualising the Polynomial Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, Y, color='red')
plt.plot(X_grid, lin_reg2.predict(poly_reg.fit_transform(X_grid)), color='blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Predict a new result with Linear Regression
bad_prediction = lin_reg.predict([[6.5]])  # we have to give as an argument an array
good_prediction = lin_reg2.predict(poly_reg.fit_transform([[6.5]]))
print(good_prediction)
print(bad_prediction)

