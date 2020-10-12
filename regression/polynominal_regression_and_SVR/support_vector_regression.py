import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values # we are taking all columns except last one
Y = dataset.iloc[:, -1].values  # values of the last column

# Feature scaling
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(Y)
