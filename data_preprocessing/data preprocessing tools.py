import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer as SI

# import data from dataset
dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:, :-1].values # we are taking all columns except last one
y = dataset.iloc[:, -1].values  # values of the last column
print(type(x))  # without .value it would be class not nd.array
print(x)
print(y)

# taking care of missing data
imputer = SI(missing_values=np.nan, strategy='mean')  # creating object, which will fulfill missing data
imputer.fit(x[:, 1:3])  # we are choosing column in which we will looking for empty data
x[:, 1:3] = imputer.transform(x[:, 1:3])  # making transform

print(x)
