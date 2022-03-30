import numpy as np
import matplotlib.pyplot as plt
import math
import random

#Make 2 sets of random data
SIZE = 500
D = 2
K = 2
class1=np.random.multivariate_normal([1,3],[[1,0],[0,1]],math.ceil(SIZE/2.0))
class2=np.random.multivariate_normal([4,1],[[2,0],[0,1]],math.floor(SIZE/2.0))
X = np.vstack((class1,class2))
Y = np.hstack((np.ones(math.ceil(SIZE/2.0)),np.ones(math.floor(SIZE/2.0))*0))

#Randomize Data Order
DataX = np.empty((0,D),float)
DataC = np.empty((0,1),int)
Remain = SIZE - 1
for i in range(SIZE):
    p = random.randint(0,Remain)
    DataX = np.append(DataX,np.array([X[p]]),0)
    DataC = np.append(DataC,Y[p])
    X = np.delete(X,p,0)
    Y = np.delete(Y,p,0)
    Remain = Remain - 1
DataX = np.append(DataX,np.ones((SIZE,1)),1)

DataT = np.empty((0,K),int)
for i in range(SIZE):
    if DataC[i] == 1:
        DataT = np.append(DataT,np.array([[1,0]]),0)
    elif DataC[i] == 0:
        DataT = np.append(DataT,np.array([[0,1]]),0)

#Gradient Descent Bitches

W = np.random.rand((D+1),K)


maxEpochs = 1000
LR = .2
for i in range(maxEpochs):
    Sumtemp = np.zeros(((D+1),K))
    for j in range(SIZE):
        temp1 = 0
        for k in range(K):
            temp1 += np.exp(np.matmul(np.transpose(W)[k],DataX[j][:]))

        for k in range(K):
            temp2 = (np.exp(np.matmul(np.transpose(W)[k],DataX[j][:])))/temp1
            temp3 = temp2 * DataX[j][:]
            np.transpose(Sumtemp)[k] = np.transpose(np.transpose(Sumtemp)[k] + temp3)
    W = W - LR * Sumtemp
    print(W)
plt.figure(1)
for i in range(len(DataX)):
    if DataC[i] == 1:
        plt.plot(DataX[i][0],DataX[i][1],'ro')
    else:
        plt.plot(DataX[i][0],DataX[i][1],'gs')

plt.xlim([-2, 8])
plt.ylim([-2, 6])
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()