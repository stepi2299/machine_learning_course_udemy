import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

dataset = pd.read_csv('../Social_Network_Ads.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

# Training the Random Forrest Classification
classifier = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0)  # we can decide how many decision trees the algorithm will build
classifier.fit(X_train, Y_train)

# Predict a new result
y_uno_pred = classifier.predict(sc.transform([[30, 8700]]))  # i have to remember to transform the data that i check if i did feature scaling earlier
print(y_uno_pred)

# Predict the Test set result
y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), Y_test.reshape(len(Y_test), 1)), 1))  # we are creating matrix or two vectors to easier compare them

# Making the Confusion Matrix
# confusion matrix is 2D matrix (2 rows, 2 columns) and it will how as number of corrected predictions in previous point
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(Y_test, y_pred)
print(cm)
a_s = accuracy_score(Y_test, y_pred)  # return a value of accuracy (1 is the highest value)
print(a_s)  # it returns how many corrected predictions we have in percentage
