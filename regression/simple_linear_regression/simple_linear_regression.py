import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values # we are taking all columns except last one
Y = dataset.iloc[:, -1].values  # values of the last column

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# training the simple linear regression model on the training set
regressor = LinearRegression()  # we are building simple linear regression
regressor.fit(X_train, Y_train)  # we are training regression model

# predict the test set result
y_pred = regressor.predict(X_test)  # it will return vector of predictions based on training for x_test

# visualing the training set result
plt.scatter(X_train, Y_train, color='red')  # making point on the diagram
plt.plot(X_train, regressor.predict(X_train), color='blue')  # making line of predicted X_train set
plt.title('Salary vs experience(Training set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()

# visualing the test set result
plt.scatter(X_test, Y_test, color='red')  # making point on the diagram
plt.plot(X_train, regressor.predict(X_train), color='blue')  # making line of predicted X_train set
plt.title('Salary vs experience(Test set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()
