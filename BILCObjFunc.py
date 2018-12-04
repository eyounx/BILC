import numpy as np
import math

def BILCObjFunc(x,data):
    n,l = data['lbl'].shape
    k = data['k']

    sample = np.array(x)
    M = sample.reshape(k,l)
    X = np.dot(M, (data['lbl'].T))
    X[X<=0]=0
    X[X>0]=1

    M_hat = np.linalg.pinv(M)
    Y_hat = np.dot(M_hat, X)

    Y_hat = Y_hat.T#n*l
    res=0
    sorted = np.sort(Y_hat,axis=1)
    for topk in [1,3,5]:
        tmpY = Y_hat.copy()
        tmpY[tmpY < sorted[:,None,-1*topk]]=0
        tmpY[tmpY!=0]=1
        res = res + (tmpY!=data['lbl']).sum() * math.exp(-1*topk)
    return res



