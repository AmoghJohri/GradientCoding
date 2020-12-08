import numpy as np 

def get_absolute_loss(X, y, w):
    loss = 0.
    for i in range(X.shape[0]):
        loss += abs(y[i] - np.dot(w, X[i]))
    return loss/X.shape[0]

def least_square_loss(X, y, w):
    return np.dot(X.T, np.subtract(np.dot(X, w), y))