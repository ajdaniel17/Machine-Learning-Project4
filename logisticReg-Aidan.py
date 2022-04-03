import numpy as np
import matplotlib.pyplot as plt
import math
import random
import time
import cv2 as cv
import idx2numpy
import scipy.special as sp

# Load Train Images and Labels Files
imageTrainFile = 'MNIST/DataUncompressed/train-images.idx3-ubyte'
labelTrainFile = 'MNIST/DataUncompressed/train-labels.idx1-ubyte'

# Convert files to numpy arrays
imageTrainArr = idx2numpy.convert_from_file(imageTrainFile)
labelTrainArr = idx2numpy.convert_from_file(labelTrainFile)

# Linearize images
imageTrainArrLinearized = imageTrainArr.reshape(imageTrainArr.shape[0], imageTrainArr.shape[1] * imageTrainArr.shape[2])

# Generate T array from labels array (One-Hot Encoding)
def generateT(N, K, labelArr):
    T = np.zeros((N, K))
    for i in range(0, labelArr.shape[0]):
        T[i, labelArr[i]] = 1
    return T

# Normalize pixel values and generate data matrix
def normalizeAndGenerateDataMatrix(imageArr):
    imageArr = imageArr / 255
    # imageArr = (imageArr - np.mean(imageArr)) / np.std(imageArr)
    return np.insert(imageArr, imageArr.shape[1], 1, axis=1)

def accuracy(W,X,T):
    SIZE , D = X.shape
    yPred = np.dot(X, W)
    mistakes = 0
    for i in range(SIZE):
        if (np.argmax(yPred[i]) != np.argmax(T[i])):
            mistakes += 1
    return ((SIZE - mistakes) / SIZE) * 100




def Gradient_Descent(DataX, DataT):
    SIZE , D = DataX.shape
    SIZE , K = DataT.shape
    Beta = .9
    PrevSumtemp = np.zeros(((D),K))
    W = np.random.rand((D),K)
    print("Starting W: ")
    print(W.shape)
    maxEpochs = 500
    LR = .5
    for i in range(maxEpochs):
        Sumtemp = np.zeros(((D),K))
        for j in range(1000):
            temp1 = 0
            for k in range(K):
                temp1 += np.exp(np.matmul(np.transpose(W)[k],DataX[j][:])- np.max(np.matmul(np.transpose(W)[k],DataX[j][:])) )

            for k in range(K):
                temp2 = (np.exp(np.matmul(np.transpose(W)[k],DataX[j][:]) - np.max(np.matmul(np.transpose(W)[k],DataX[j][:]))  ))/temp1
                temp3 = (temp2 - DataT[j][k]) * DataX[j][:]
                np.transpose(Sumtemp)[k] = np.transpose(np.transpose(Sumtemp)[k] + temp3)
        #print(Sumtemp)

        Sumtemp = Beta*PrevSumtemp + (1.0 - Beta) * Sumtemp
        W = W - LR * Sumtemp
        PrevSumtemp = Sumtemp
        print("Epoch ", i , "Accuracy:", accuracy(W,DataX,DataT) )
        #print(W)
    return W


# Number of Classes
K = 10
# Number of Images
N = imageTrainArrLinearized.shape[0]
# Number of Features
M = imageTrainArrLinearized.shape[1]
T = generateT(N, K, labelTrainArr)
imageDataMatrix = normalizeAndGenerateDataMatrix(imageTrainArrLinearized)
trainedModel = Gradient_Descent(imageDataMatrix, T)

