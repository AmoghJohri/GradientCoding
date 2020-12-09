import random
import statistics
import numpy as np 
import matplotlib.pyplot as plt
from numpy.lib.function_base import average, diff 

class State:
    def __init__(self, curr, ps, pns):
        self.curr = curr 
        self.pss = ps 
        self.pns = pns

    def iter(self):
        r = np.random.random()
        if self.curr == 0:
            if r > self.pss:
                self.curr = 1 # corresponds to straggler
        else:
            if r > self.pns:
                self.curr = 0 # corresponds to non-straggler

class System:
    def __init__(self, n, r, s, init, pss, pns):
        self.pss = pss
        self.pns = pns
        self.n = n
        self.s = s 
        self.r = r 
        self.nBins = int(n/s)
        self.nodes = []
        for i in range(0, n):
            if i in init:
                self.nodes.append(State(1, self.pss, self.pns))
            else:
                self.nodes.append(State(0, self.pss, self.pns))
        
    def frc(self, it, shuffle=0):
        for each in self.nodes:
            each.iter()
        data = [0 for i in range(self.nBins)]
        for j in range(0, self.n):
            if self.nodes[j].curr == 0:
                i = (j+it*shuffle)%self.n
                data[i//self.s] = 1
        return np.asarray(data)

    def crc(self, it, shuffle=0):
        for each in self.nodes:
            each.iter()
        data = [0 for i in range(self.n)]
        for j in range(0, self.n):
            if self.nodes[j].curr == 0:
                i = (j+it*shuffle)%self.n
                for k in range(self.s):
                    data[(i+k)%self.n] = 1
        return np.asarray(data)

def sim_frc(n, k, pss, pns, r_percentage, iterations, shuffle=0):
    r = int(n*r_percentage)
    data  = np.asarray([0 for i in range(int(n/k))])
    model = System(n, r, k, [random.randint(0, n-1) for i in range(n-r)], pss, pns)
    for i in range(iterations):
        data += model.frc(i, shuffle=shuffle)
    data = data.tolist()
    plt.bar(["D"+str(i) for i in range(int(n/k))], data)
    plt.bar(["MAX"], [iterations])
    plt.grid()
    plt.xlabel("Data Bucket")
    plt.ylabel("Data Received")
    if shuffle == 1:
        plt.title("FRC - With Shuffling")
    else:
        plt.title("FRC")
    plt.show()

def sim_crc(n, k, pss, pns, r_percentage, iterations, shuffle=0):
    r = int(n*r_percentage)
    data  = np.asarray([0 for i in range(int(n))])
    model = System(n, r, k, [random.randint(0, n-1) for i in range(n-r)], pss, pns)
    for i in range(iterations):
        data += model.crc(i, shuffle=shuffle)
    data = data.tolist()
    plt.bar(["D"+str(i) for i in range(int(n))], data)
    plt.bar(["MAX"], [iterations])
    plt.grid()
    plt.xlabel("Data Bucket")
    plt.ylabel("Data Received")
    if shuffle == 1:
        plt.title("CRC - With Shuffling")
    else:
        plt.title("CRC")
    plt.show()

if __name__ == "__main__":
    n            = 36
    k            = 2
    pss          = 0.95
    pns          = 0.95
    r_percentage = 0.33
    iterations   = 1000
    sim_crc(n, k, pss, pns, r_percentage, iterations, shuffle=1)
    

