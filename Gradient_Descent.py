import numpy as np
import matplotlib.pyplot as plt
import math
import random
import time

def Gradient_Descent(DataX,DataT):
    start_time = time.time()
    SIZE, D = DataX.shape
    SIZE, K = DataT.shape

    W = np.random.rand((D+1),K)
    maxEpochs = 3000
    NE = 0
    LR = .002
    for i in range(maxEpochs):
        Sumtemp = np.zeros(((D+1),K))
        for j in range(SIZE):
            temp1 = 0
            for k in range(K):
                temp1 += np.exp(np.matmul(np.transpose(W)[k],DataX[j][:]))

            for k in range(K):
                temp2 = (np.exp(np.matmul(np.transpose(W)[k],DataX[j][:])))/temp1
                temp3 = (temp2 - DataT[j][k]) * DataX[j][:]
                np.transpose(Sumtemp)[k] = np.transpose(np.transpose(Sumtemp)[k] + temp3)

        #loss = -np.sum(np.sum(DataT, axis=0) * np.log(np.sum(temp2, axis=0))) / SIZE
        W = W - LR * Sumtemp
        NE += 1
    total_time = time.time() - start_time 
    return W, NE,total_time