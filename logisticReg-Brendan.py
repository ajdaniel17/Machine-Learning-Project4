import time
import cv2 as cv
import numpy as np
import idx2numpy
import matplotlib.pyplot as plt
import scipy.special as sp

# np.set_printoptions(threshold=np.inf)

# Load Train Images and Labels Files
imageTrainFile = 'MNIST/DataUncompressed/train-images.idx3-ubyte'
labelTrainFile = 'MNIST/DataUncompressed/train-labels.idx1-ubyte'

# Convert files to numpy arrays
imageTrainArr = idx2numpy.convert_from_file(imageTrainFile)
labelTrainArr = idx2numpy.convert_from_file(labelTrainFile)

# Linearize images
imageTrainArrLinearized = imageTrainArr.reshape(imageTrainArr.shape[0], imageTrainArr.shape[1] * imageTrainArr.shape[2])

def softmax(z):
    # z --> linear
    exp = np.exp(z-np.max(z))

    for i in range(len(z)):
        exp[i] /= np.sum(exp[i])

    return exp

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

# IN PROGRESS
def Gradient_Descent(DataX, DataT):
    start_time = time.time()
    SIZE, D = DataX.shape
    SIZE, K = DataT.shape
    W = np.random.rand((D),K)
    maxEpochs = 1000
    NE = 0
    LR = .5
    for i in range(maxEpochs):
        temp1 = sp.softmax(np.dot(DataX, W))
        gradient = np.dot(np.transpose(DataX), temp1) - np.dot(np.transpose(DataX), DataT)
        W = W - LR * gradient
        NE += 1
        if (np.linalg.norm(sp.softmax(np.dot(DataX, W)) - temp1) < 1e-8):
            print(NE)
            break
    total_time = time.time() - start_time 
    return W, NE, total_time

# def fitModel(M, K, N, dataMatrix, T):
#     W = np.zeros(((M + 1), K))
#     V = np.zeros(((M + 1), K))
#     beta = 0.9
#     learningRate = 0.05
#     # Iterations
#     for i in range(0, 1000):
#         # Images
#         for j in range(0, N):
#             currentX = dataMatrix[j, :]
#             currentT = T[j, :]
#             gradient = (((W.T * currentX).T - currentT).T * currentX).T
#             V = beta * V + (1 - beta) * gradient
#             W = W - learningRate * V
#             # print(W)
#         # print("Iteration Done")
#         # input('?')
#         # print("W\n", W)
#         yPred = sp.softmax(np.dot(dataMatrix, W))
#         #print(yPred.shape)
#         loss = -np.sum(np.sum(T, axis=0) * np.log(np.sum(yPred, axis=0))) / 60000
#         #loss = np.sum(-np.sum(T*np.log(yPred), axis=0)) / 60000
#         print(i)
#         print(loss)
        

# Number of Classes
K = 10
# Number of Images
N = imageTrainArrLinearized.shape[0]
# Number of Features
M = imageTrainArrLinearized.shape[1]
T = generateT(N, K, labelTrainArr)
imageDataMatrix = normalizeAndGenerateDataMatrix(imageTrainArrLinearized)
trainedModel, numIter, totalTime = Gradient_Descent(imageDataMatrix, T)
yPred = np.dot(imageDataMatrix, trainedModel)
mistakes = 0
for i in range(N):
    if (np.argmax(yPred[i]) != np.argmax(T[i])):
        mistakes += 1

print((N - mistakes) / N)
