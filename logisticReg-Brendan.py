from email.mime import image
import cv2 as cv
import numpy as np
import idx2numpy
import matplotlib.pyplot as plt
import scipy.special as sp

# Load Train Images and Labels Files
imageTrainFile = 'MNIST/DataUncompressed/train-images.idx3-ubyte'
labelTrainFile = 'MNIST/DataUncompressed/train-labels.idx1-ubyte'

# Convert files to numpy arrays
imageTrainArr = idx2numpy.convert_from_file(imageTrainFile)
labelTrainArr = idx2numpy.convert_from_file(labelTrainFile)

# Linearize images
imageTrainArrLinearized = imageTrainArr.reshape(imageTrainArr.shape[0], imageTrainArr.shape[1] * imageTrainArr.shape[2])

# Generate T array from labels array
def generateT(N, K, labelArr):
    T = np.zeros((N, K))
    for i in range(0, labelArr.shape[0]):
        T[i, labelArr[i]] = 1
    return T

# Normalize pixel values and generate data matrix
def normalizeAndGenerateDataMatrix(imageArr):
    imageArr = imageArr / 255
    imageArr = (imageArr - np.mean(imageArr)) / np.std(imageArr)
    return np.insert(imageArr, imageArr.shape[1], 1, axis=1)

# IN PROGRESS
def fitModel(M, K, N, dataMatrix, T):
    W = np.random.random(((M + 1), K))
    V = np.zeros(((M + 1), K))
    print(dataMatrix.shape)
    print(T.shape)
    for i in range(0, 1000):
        for j in range(0, N):
            dataMatrix = dataMatrix[j, :]
            T = T[j, :]
            gradient = np.dot(np.dot(W.T, dataMatrix), dataMatrix) - np.dot(T, dataMatrix)
            print(gradient.shape)
        # for j in range(0, N):
        #     currentX = dataMatrix[j, :]
        #     currentT = T[j, :]
        #     print(sp.softmax(W.T * currentX).shape)
        #     print(currentT.shape)
        #     gradient = ((W.T * currentX) - currentT) * currentX
        # print(gradient.shape)

# Number of Classes
K = 10
# Number of Images
N = imageTrainArrLinearized.shape[0]
# Number of Features
M = imageTrainArrLinearized.shape[1]
T = generateT(N, K, labelTrainArr)
imageDataMatrix = normalizeAndGenerateDataMatrix(imageTrainArrLinearized)
fitModel(M, K, N, imageDataMatrix, T)



