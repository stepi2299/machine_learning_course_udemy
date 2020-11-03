import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

dataset = pd.read_excel('Folds5x2_pp.xlsx')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Building ANN
ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

# adding the output layer (interesting is which activation function)
ann.add(tf.keras.layers.Dense(units=1))  # in regression (so one continous output without activation function)

# Training the ANN
ann.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
ann.fit(X_train, y_train,batch_size=32, epochs=100)

# Predictions
y_pred = ann.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))
