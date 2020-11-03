import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

# Preprocessing data
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])
print(X)

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')  # we are transforming
X = np.array(ct.fit_transform(X))  # we are connecting and transforming our np.array
print(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

sc = StandardScaler()  # creating object of standard scaler
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
print(X_train)
print(X_test)

# Building the ANN
# Initialize the ANN as sequence of layers
ann = tf.keras.models.Sequential()  # creating ANN as a sequence of layers

# adding the input layer and the first hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))  # parameter units tell us how many neurons will be in a layer, the first hidden layer

# adding the second hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

# Adding the output layer
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))  # the output should be binary so units are equal to 1

# Training the ANN
# compiling the ANN
ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  # very important parameters

# training the ANN on the Training set
ann.fit(X_train, y_train, batch_size=32, epochs=100)

# Making the predictions and evaluating the model
# Predict the result of a single observations
print(ann.predict(sc.fit_transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])) > 0.5)

# Predict the Test set reslut
y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)  # making the binary outcome
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

# making the confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
a_s = accuracy_score(y_test, y_pred)  # return a value of accuracy (1 is the highest value)
print(a_s)  # it returns how many corrected predictions we have in percentage
