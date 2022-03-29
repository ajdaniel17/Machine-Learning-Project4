import numpy as np
import matplotlib.pyplot as plt

DataX = np.array([[1,1],
                  [1,-1],
                  [2,1],
                  [3,1],
                  [3,-1],
                  [4,1]])

DataC = np.array([[1,0],
                  [1,0],
                  [1,0],
                  [0,1],
                  [0,1],
                  [0,1]])

plt.figure(1)
for i in range(len(DataX)):
    if DataC[i][0] == 1:
        plt.plot(DataX[i][0],DataX[i][1],'ro')
    else:
        plt.plot(DataX[i][0],DataX[i][1],'gs')

plt.xlim([-3, 7])
plt.ylim([-3, 3])
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()