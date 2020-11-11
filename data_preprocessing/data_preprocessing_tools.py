# Data processing tools

# importing the most important libraries
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
X = dataset.iloc[:, :-1].values  # grabbing all columns except last one (all independent variables)
y = dataset.iloc[:, -1].values  # values of the last column (all dependent variables

# taking care of missing data
# we are using this tool when our dataset has some missing places
imputer = SI(missing_values=np.nan, strategy='mean')  # creating object, which will train our dataset to fulfill empty space
imputer.fit(X[:, 1:3])  # we are connecting column in which we will looking for empty data
X[:, 1:3] = imputer.transform(X[:, 1:3])  # making transform
print(X)

# Encoding categorical data
# France as [1,0,0], Germany = [0,1,0], Spain = [0,0,1] -> OneHotEncode
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')  # we are transforming
X = np.array(ct.fit_transform(X))  # we are changing for example string value into dummy variables and transforming into np.array

# convert Yes into 1 and No as 0
le = LabelEncoder()
y = le.fit_transform(y)

# Splitting the dataset into the Training set and Test set
# Test set will be 20% of our current dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature scaling
# Remember to change only numeric variables (not dummy, binary etc.)
sc = StandardScaler()  # creating object of standard scaler which will convert our chosen data into range (-3, 3)
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc.transform(X_test[:, 3:])

