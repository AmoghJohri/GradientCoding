import random
import numpy             as np
import matplotlib.pyplot as plt 

class Model:
    def __init__(self, n, r, k, pss, pns):
        """
        n   :: number of compute instances
        r   :: number of non-stragglers
        k   :: repetition (for fractional repetition scheme)
        pss :: probability of stragglers to straggle
        pns :: probability of non-stragglers to straggle 
        """
        self.n          = n 
        self.r          = r 
        self.k          = k
        self.bins       = int(n/k) 
        self.pss        = pss 
        self.pns        = pns 
        self.stragglers = [random.randint(0, self.n-1) for i in range(self.n-self.r)]

    def frc(self, it, shuffle=0):
        it   = it * shuffle 
        data = [0 for i in range(self.bins)]
        for i in range(self.n):
            if i in self.stragglers:
                p = self.pss 
            else:
                p = self.pns 
            if np.random.uniform() > p:
                data[int(((i + it)%self.n)//self.k)] = 1
        return np.asarray(data)
    
    def crc(self, it, shuffle=0):
        it = it * shuffle 
        data = [0 for i in range(self.n)]
        for i in range(self.n):
            if i in self.stragglers:
                p = self.pss 
            else:
                p = self.pns 
            if np.random.uniform() > p:
                i = (i + it)%self.n
                for k in range(self.k):
                    data[int(((i + k)%self.n))] = 1
        return np.asarray(data)

def sim_frc(n, k, pss, pns, r_percentage, iterations, shuffle=0):
    r = int(n*r_percentage)
    data  = np.asarray([0 for i in range(int(n/k))])
    model = Model(n, r, k, pss, pns)
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
    model = Model(n, r, k, pss, pns)
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
    n            = 72
    s_list       = [2, 3, 4, 6, 8, 9, 12, 18]
    pss          = 0.80 
    pns          = 0.01
    r_percentage = 0.33
    iterations   = 1000
    for s in s_list:
        sim_crc(int(n/2), s, pss, pns, r_percentage, iterations, shuffle=1)
    
    
    
