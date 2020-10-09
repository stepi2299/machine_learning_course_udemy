import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer as SI
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# import data from dataset
dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:, :-1].values # we are taking all columns except last one
y = dataset.iloc[:, -1].values  # values of the last column
print(type(x))  # without .value it would be class not nd.array
print(x)
print(y)

# taking care of missing data
imputer = SI(missing_values=np.nan, strategy='mean')  # creating object, which will fulfill missing data
imputer.fit(x[:, 1:3])  # we are connecting column in which we will looking for empty data
x[:, 1:3] = imputer.transform(x[:, 1:3])  # making transform

print(x)

# Encoding categorical data
# France as [1,0,0], Germany = [0,1,0], Spain = [0,0,1] -> OneHotEncode
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')  # we are transforming
x = np.array(ct.fit_transform(x))  # we are connecting and transforming our np.array
print(x)

# convert Yes into 1 and No as 0
le = LabelEncoder()
y = le.fit_transform(y)
print(y)

# Splitting the dataset into the Training set and Test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
print(x_train)
print(x_test)
print(y_train)
print(y_test)

# Feature scaling
sc = StandardScaler()  # creating object of standard scaler
x_train[:, 3:] = sc.fit_transform(x_train[:, 3:])
x_test[:, 3:] = sc.transform(x_test[:, 3:])
print(x_train)
print(x_test)
