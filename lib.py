import numpy as np
import config
import lib

def relu(t):
    return np.maximum(t, 0)

def softmax(t):
    return np.exp(t) / np.sum(out, axis=1, keepdims=True)

# def predict(x):
    

def sparse_cross_entropy(z, y):
    return -np.log(np.array(z[j, y[j] for j in range(len(y))]))

def to_full(y, num_classes):
    y_full = np.zeros((1, num_classes))
    y_full[0, y] = 1
    return y_full

def relu_deriv(t):
    return (t >= 0).astype(float)

