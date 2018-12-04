from Racos import RacosOptimization
from Components import Dimension
from BILCObjFunc import BILCObjFunc
from BILC import PrecisionAtK
from BILC import TrainMultiClassifier
import numpy as np
import h5py


def ResultAnalysis(res, top):
    res.sort()
    top_k = []
    for i in range(top):
        top_k.append(res[i])
    mean_r = np.mean(top_k)
    std_r = np.std(top_k)
    print mean_r, '#', std_r
    return

def loadmat(filename):
    data = h5py.File(filename)
    trn_ft = np.array(data['trn_ft']).T
    trn_lbl = np.array(data['trn_lbl']).T
    tst_ft = np.array(data['tst_ft']).T
    tst_lbl = np.array(data['tst_lbl']).T
    return trn_ft,trn_lbl,tst_ft,tst_lbl


# parameters
SampleSize = 20             # the instance number of sampling in an iteration
MaxIteration = 800000       # the number of iterations
Budget = 2000               # budget in online style
PositiveNum = 1             # the set size of PosPop
RandProbability = 0.99      # the probability of sample in model
UncertainBits = 1           # the dimension size that is sampled randomly
k = 5                         # the embedding dimension

# load data
trn_ft, trn_lbl, tst_ft, tst_lbl = loadmat('Data/core5k_kfold1.mat')
n,l = trn_lbl.shape

# dimension setting
results = []
DimSize = k*l
regs = []
regs.append(-1.0)
regs.append(1.0)

dim = Dimension()
dim.setDimensionSize(DimSize)
for i in range(DimSize):
    dim.setRegion(i, regs, True)

# data process
data = {'ft':trn_ft, 'lbl':trn_lbl, 'k':k}

# Racos get M and M_hat
print i, ':--------------------------------------------------------------'
racos = RacosOptimization(dim)

# call online version RACOS
# racos.OnlineTurnOn()
# racos.ContinueOpt(Ackley, SampleSize, Budget, PositiveNum, RandProbability, UncertainBits)

racos.ContinueOpt(BILCObjFunc, SampleSize, MaxIteration, PositiveNum, RandProbability, UncertainBits, data)

# print racos.getOptimal().getFeatures()
print racos.getOptimal().getFitness()

sample = np.array(racos.getOptimal().getFeatures())
M = sample.reshape(k, l)
M_hat = np.linalg.pinv(M)

# Train
ClassifierArg={'max_depth':2,
               'n_estimators':250,
               'learning_rate':0.5}

pred_tst_score = TrainMultiClassifier(trn_ft,trn_lbl,tst_ft,tst_lbl,M,M_hat,k,ClassifierArg)
p1,p3,p5,avgP = PrecisionAtK(tst_lbl,pred_tst_score)

print(p1,p3,p5,avgP)
