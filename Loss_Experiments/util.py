import scipy.special as sp
import numpy as np 
import itertools

def get_absolute_loss(X, y, w):
    loss = 0.
    for i in range(X.shape[0]):
        loss += abs(y[i] - np.dot(w, X[i]))
    return loss/X.shape[0]

def least_square_loss(X, y, w):
    return np.dot(X.T, np.subtract(np.dot(X, w), y))

def unique_rows(a):
    b = np.ascontiguousarray(a).view(np.dtype((np.void, a.dtype.itemsize * a.shape[1])))
    _, idx = np.unique(b, return_index=True)
    return a[idx]

def getB(n_workers,n_stragglers):
    Htemp=np.random.normal(0,1,[n_stragglers,n_workers-1])
    H=np.vstack([Htemp.T,-np.sum(Htemp,axis=1)]).T
    Ssets=np.zeros([n_workers,n_stragglers+1])
    for i in range(n_workers):
        Ssets[i,:]=np.arange(i,i+n_stragglers+1)
    Ssets=Ssets.astype(int)
    Ssets=Ssets%n_workers
    B=np.zeros([n_workers,n_workers])
    for i in range(n_workers):
        B[i,Ssets[i,0]]=1
        vtemp=-np.linalg.solve(H[:,np.array(Ssets[i,1:])],H[:,Ssets[i,0]])
        ctr=0
        for j in Ssets[i,1:]:
            B[i,j]=vtemp[ctr]
            ctr+=1
    return B

def getA(B,n_workers,n_stragglers):
    S = np.ones((int(sp.binom(n_workers,n_stragglers)),n_workers))
    combs = itertools.combinations(range(n_workers), n_stragglers)
    i=0
    for pos in combs:
        S[i,pos] = 0
        i += 1
    (m,n)=S.shape
    A=np.zeros([m,n])
    for i in range(m):
        sp_pos=S[i,:]==1
        A[i,sp_pos]=np.linalg.lstsq(B[sp_pos,:].T,np.ones(n_workers))[0]
    return A