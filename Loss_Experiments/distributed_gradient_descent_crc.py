import time
import numpy             as np
import matplotlib.pyplot as plt 
from data_generator import get_dataset
from util           import get_absolute_loss
from util           import least_square_loss

def merge(arr):
    out = []
    for each in arr:
        for _ in each:
            out.append(_)
    return out 

def iterate(f, X, y, w, bucket, alpha=0.0001):
    s = bucket[0]
    e = bucket[1]
    return np.multiply((alpha/X.shape[0]), f(X[s:e], y[s:e], w))

def run(N=10, s_ratio=0.1, pss=0.8, pns=0.01, k=2, time_stop=0, dataset="boston", alpha=0.0001, adversarial=False):
    data               = get_dataset(dataset)
    X_, y              = data.data, data.target 
    X                  = np.ones((X_.shape[0], X_.shape[1] + 1))
    X[:,:-1]           = X_  
    X                  = np.divide((X - np.mean(X)),np.std(X))
    if not adversarial:
        np.random.seed(42)
        np.random.shuffle(X)
        np.random.seed(42)
        np.random.shuffle(y)
        X_test             = X[:int(X.shape[0]/2)]
        X                  = X[int(X.shape[0]/2):X.shape[0]]
        y_test             = y[:int(y.size/2)]
        y                  = y[int(y.size/2):y.size]
    else:
        indices            = y.argsort()
        y_sorted           = y[indices[::-1]]
        X_sorted           = X[indices[::-1]]
        if ((np.unique(y)).size) <= 10:
            train_size = int((y.size/np.unique(y).size)/2)
            X_train            = []
            y_train            = []
            X_test             = []
            y_test             = []
            counter            = 0
            for i in range(y.size):
                if counter < train_size:
                    X_train.append(X_sorted[counter])
                    y_train.append(y_sorted[counter])
                else:
                    X_test.append(X_sorted[counter])
                    y_test.append(y_sorted[counter])
                counter += 1
                if counter > 0 and y_sorted[counter-1] != y_sorted[counter]:
                    counter = 0
            X                  = X_train 
            y                  = y_train 
        else:
            X_test             = X
            y_test             = y
            X                  = X_sorted 
            y                  = y_sorted
        X      = np.asarray(X)
        y      = np.asarray(y)
        X_test = np.asarray(X_test)
        y_test = np.asarray(y_test)
    np.random.seed(0)
    w                  = np.random.rand(X.shape[1])
    loss_arr           = [get_absolute_loss(X_test, y_test, w)]
    time_arr           = [0.]
    units              = X.shape[0]/(N)
    buckets            = []
    start              = 0
    for i in range(N-1):
        buckets.append((int(start), int(start + units*k)))
        start = start + units
    s                  = int(N*s_ratio)
    straggler_arr      = np.random.randint(low = 0, high = len(buckets)-1, size = s)
    t_max_factor       = 2
    i = 0
    while True:
        g     = 0
        t     = 0
        k_arr = []
        for each in buckets:
            SD = 1
            start_time = time.time()
            g_         = iterate(least_square_loss, X, y, w, each, alpha=alpha)
            t_         = time.time() - start_time 
            if buckets.index(each) in straggler_arr:
                if np.random.random() < pss:
                    SD = np.random.normal(loc=9.0, scale=3.0)
            else:
                if np.random.random() < pns:
                    SD = np.random.normal(loc=9.0, scale=3.0)
            t_    = t_*SD
            k_arr.append(t_)
            if buckets.index(each) != 0 and buckets.index(each) % k == 0:
                t_ = min(k_arr)
                k_arr = [0 for j in range(k)]
                if t_ < t_max_factor:
                    g += g_
                    if t == 0 or t_ > t:
                        t = t_
        w = np.subtract(w, g)
        loss_arr.append(get_absolute_loss(X_test, y_test, w))
        time_arr.append(time_arr[-1] + t)
        i += 1
        if time_stop != 0:
            if time_arr[-1] > time_stop:
                break 
        else:
            if i == 1000:
                break
    return time_arr, loss_arr 

if __name__ == "__main__":
    time_arr, loss_arr = run()
    plt.plot(time_arr, loss_arr)
    plt.xlabel("Time (s)")
    plt.ylabel("Absolute Loss")
    plt.title("Loss Curve")
    plt.grid()
    plt.show()