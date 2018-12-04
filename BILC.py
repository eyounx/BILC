import numpy as np
import math
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

def PrecisionAtK(target, score):
    n,l = target.shape
    sorted = np.sort(score,axis=1)
    p=[]
    avgP=0
    for topk in [1,3,5]:
        tmp = score.copy()
        tmp[tmp < sorted[:, None, -1 * topk]] = 0
        tmp[tmp != 0] = 1
        diff = target[tmp==1]
        pk = (diff==1).sum()/(n*topk+0.0)
        p.append(pk)
        avgP = avgP + pk * math.exp(-1*topk)
    return p[0],p[1],p[2],avgP

def TrainMultiClassifier(trn_ft,trn_lbl,tst_ft,tst_lbl,M,M_hat,k,ClassifierArg):
    X = np.dot(M, trn_lbl.T)  # k*n
    X[X <= 0] = 0
    X[X > 0] = 1
    predX = np.zeros([tst_lbl.shape[0], k])
    for i in range(0, k):
        bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=ClassifierArg['max_depth']),
                                 n_estimators=ClassifierArg['n_estimators'],
                                 learning_rate=ClassifierArg['learning_rate'])
        bdt.fit(trn_ft, X[i, :])
        tmp = bdt.predict(tst_ft)
        predX[:, i] = tmp

    pred_tst_score = np.dot(M_hat, predX.T)  # l*n
    return pred_tst_score.T