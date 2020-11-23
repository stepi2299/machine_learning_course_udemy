import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_classification
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


# data generation
m = 300
data = make_classification(n_samples=m, n_features=4, n_informative=3, n_redundant=0, n_classes=3, random_state=0)
X = data[0]
X = X.transpose()
y = data[1]
y = y.reshape(m, 1)
print(y)
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')  # we are transforming
y = np.array(ct.fit_transform(y))
y.
print(y)

# initialization of weights
np.random.seed(0)

def initialize():
    W1 = np.random.normal(0, 0.1, size=(6, 4))
    b1 = np.zeros(shape=(6, 1))
    W2 = np.random.normal(0, 0.1, size=(6, 6))
    b2 = np.zeros(shape=(6, 1))
    W3 = np.random.normal(0, 0.1, size=(3, 6))
    b3 = np.zeros(shape=(3, 1))
    return W1, b1, W2, b2, W3, b3

# Activation functions
def sigmoid(z):
    return 1/(1+np.exp(-z))

def leaky_relu(z):
    return np.where(z > 0, z, 0.01*z)

def leaky_relu_grad(z):
    return np.where(z < 0, 0.01, 1.0)

def relu(z):
    return np.where(z > 0, z, 0)

def relu_grad(z):
    return np.where(z > 0, 1.0, 0.0)

def tanh(z):
    return (np.exp(z) - np.exp(-z))/(np.exp(z)+np.exp(-z))

def tanh_grad(z):
    return 1-tanh(z)**2

def softmax(z):
    return np.exp(z)/np.sum(np.exp(z), axis=0)

# Forward propagation

def forward(X, W1, W2, W3, b1, b2, b3):
    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = relu(Z2)
    Z3 = np.dot(W3, A2) + b3
    A3 = softmax(Z3)
    y_hat = A3
    return Z1, A1, Z2, A2, Z3, A3, y_hat

def J(y, y_hat):
    return -sum(y[i]*np.log(y_hat[i]))


# Backprojecting
def calculate_gradient(y, a3, a2, a1, w3, w2, w1, b2, b1, X, m):
    dZ3 = a3 - y
    dW3 = (1/m)*np.dot(dZ3, a2.transpose())
    db3 = (1/m)*np.sum(dZ3, axis=1, keepdims=True)
    dZ2 = np.dot(w3.transpose(), dZ3) * relu_grad(np.dot(w2, a1) + b2)
    dW2 = np.dot(dZ2, a1.transpose())
    db2 = (1/m)*np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.dot(w2.transpose(), dZ2)*relu_grad(np.dot(w1, X) + b1)
    dW1 = np.dot(dZ1, X.transpose())
    db1 = (1/m)*np.sum(dZ1, axis=1, keepdims=True)
    return dW3, db3, dW2, db2, dW1, db1

# gradient Decent
def update(w3, w2, w1, b3, b2, b1, dw3, dw2, dw1, db3, db2, db1, alpha=0.001):
    w3 = w3 - alpha * dw3
    w2 = w2 - alpha * dw2
    w1 = w1 - alpha * dw1
    b3 = b3 - alpha * db3
    b2 = b2 - alpha * db2
    b1 = b1 - alpha * db1
    return w3, w2, w1, b3, b2, b1

def accuracy(y, y_hat):
    j = 0
    for i in range(m):
        lok_max = 0
        c_max = 0
        for c in range(3):
            if y_hat[c][i] > lok_max:
                lok_max = y_hat[c][i]
                c_max = c
        if c_max == y[0][i]:
            j += 1
    return j/m * 100

# Training network
W1, b1, W2, b2, W3, b3 = initialize()
J_history = []
acc_history = []
for i in range(1000):
    Z1, A1, Z2, A2, Z3, A3, y_hat = forward(X, W1, W2, W3, b1, b2, b3)
    J_history.append(J(y, y_hat))
    acc_history.append(accuracy(y, y_hat))
    dW3, db3, dW2, db2, dW1, db1 = calculate_gradient(y, A3, A2, A1, W3, W2, W1, b2, b1, X, m)
    w3, w2, w1, b3, b2, b1 = update(W3, W2, W1, b3, b2, b1, dW3, dW2, dW1, db3, db2, db1)

plt.plot(J_history)
plt.title('Funkcja kosztu w zal od iter')
plt.show()

plt.plot(acc_history)
plt.title('dokładność w zal od iter')
plt.show()

print(acc_history[-1])
