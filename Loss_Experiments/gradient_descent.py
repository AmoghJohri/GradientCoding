from ast import get_source_segment
import numpy             as np
import matplotlib.pyplot as plt 
from data_generator import get_dataset
from util           import get_absolute_loss
from util           import least_square_loss

def iterate(f, X, y, w, alpha=0.001):
    return np.subtract(w, np.multiply((alpha/X.shape[0]), f(X, y, w)))

def run():
    data      = get_dataset()
    X_, y     = data.data, data.target 
    X         = np.ones((X_.shape[0], X_.shape[1] + 1))
    X[:,:-1]  = X_  
    X         = np.divide((X - np.mean(X)),np.std(X))
    np.random.seed(42)
    np.random.shuffle(X)
    np.random.seed(42)
    np.random.shuffle(y)
    w         = np.random.rand(X.shape[1])
    loss_arr  = [get_absolute_loss(X, y, w)]
    for i in range(1000):
        w = iterate(least_square_loss, X, y, w)
        loss_arr.append(get_absolute_loss(X, y, w))
    return loss_arr 

if __name__ == "__main__":
    loss_arr = run()
    plt.plot([i for i in range(len(loss_arr))], loss_arr)
    plt.xlabel("Iterations")
    plt.ylabel("Absolute Loss")
    plt.title("Loss Curve")
    plt.grid()
    plt.show()