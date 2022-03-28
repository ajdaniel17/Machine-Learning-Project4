import cv2 as cv
import numpy as np
import idx2numpy
import matplotlib.pyplot as plt
import scipy as sp

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
    print(T.shape)
    return T

# Normalize pixel values and generate data matrix
def normalizeAndGenerateDataMatrix(imageArr):
    imageArr = imageArr / 255
    return np.insert(imageArr, imageArr.shape[1], 1, axis=1)

# IN PROGRESS
def fitModel(M, K, dataMatrix, T):
    W = np.random.random(((M + 1), K))
    lossValuesArr = []

    for i in range(1000):
        gradient = np.dot(np.dot(W.T, dataMatrix.T), dataMatrix) - np.dot(T.T, dataMatrix)
        diff = .00000000000001 * gradient.T
        prevW = W
        newW = prevW - diff
        W = newW
        if (np.linalg.norm(newW - prevW) < 1e-5):
            print('broke')
            break
    print(W)

# Number of Classes
K = 10
# Number of Images
N = imageTrainArrLinearized.shape[0]
# Number of Features
M = imageTrainArrLinearized.shape[1]
T = generateT(N, K, labelTrainArr)
imageDataMatrix = normalizeAndGenerateDataMatrix(imageTrainArrLinearized)
fitModel(M, K, imageDataMatrix, T)



