import numpy as np 
import matplotlib.pyplot as plt

def factorial(n):
    out = 1
    for i in range(1, n+1):
        out *= i 
    return out

def combination(n, m):
    return factorial(n)/(factorial(m)*factorial(n-m))

def probability(N, m, p_hat, pss, pns, T):
    out = 0.
    for i in range(0, T+1):
        p1 = (pss**i * pns**(T - i))
        p2 = (1. - (pss**i * pns**(T - i)))
        out += combination(T, i) * p_hat**i * (1. - p_hat)**(T-i) * p1**m * p2**(N-m)
    return combination(N, m) * out 

def expectation(N, p_hat, pss, pns, T):
    out = 0
    for m in range(0, N+1):
        out += m*probability(N, m, p_hat, pss, pns, T)
    return out


if __name__ == "__main__":
    N = 100
    p_hat = 0.33
    pss = 0.8
    pns = 0.01 
    T = 2
    for p in range(1, 10):
        prob = []
        iterations = []
        p_hat = p*0.1
        for m in range(N+1):
            prob.append(probability(N, m, p_hat, pss, pns, T))
            iterations.append(m)
        plt.plot(iterations, prob)
    plt.legend(["p-hat: " + str(0.1*i)[:3] for i in range(1, 10)])
    plt.xlabel("Iterations")
    plt.ylabel("Probability")
    plt.title("Probability of Missing Iterations")
    plt.grid()
    plt.show()