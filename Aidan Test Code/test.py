import numpy as np
import matplotlib.pyplot as plt
import math
import random

SIZE = 500
D = 2
class1=np.random.multivariate_normal([1,3],[[1,0],[0,1]],math.ceil(SIZE/2.0))
class2=np.random.multivariate_normal([4,1],[[2,0],[0,1]],math.floor(SIZE/2.0))
X = np.vstack((class1,class2))
Y = np.hstack((np.ones(math.ceil(SIZE/2.0)),np.ones(math.floor(SIZE/2.0))*-1))


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