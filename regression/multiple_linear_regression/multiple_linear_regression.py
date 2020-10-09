import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression


dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values # we are taking all columns except last one
Y = dataset.iloc[:, -1].values  # values of the last column


# Encoding categorical data
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')  # we are transforming
X = np.array(ct.fit_transform(X))  # we are connecting and transforming our np.array
print(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Training the Multiple Linear Regression model on the Training set
regressor = LinearRegression()  # this class can be used both for simple and multiple linear regression
regressor.fit(X_train, Y_train)

# Predict the Test set result
# we will display 2 vectors
y_pred = regressor.predict(X_test)  # vector of predicted result
np.set_printoptions(precision=2)  # it will display any numerical value with only 2 decimals after coma
print(np.concatenate((y_pred.reshape(len(y_pred), 1), Y_test.reshape(len(Y_test), 1)), 1))
# we are concatenating two vectors vertically
