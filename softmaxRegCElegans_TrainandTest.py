import numpy as np
import matplotlib.pyplot as plt
import math
import time
import scipy.special as sp

def shuffleData(DataX, DataT):
    p = np.random.permutation(DataX.shape[0])
    return DataX[p], DataT[p]

def accuracy(W,X,T):
    SIZE , D = X.shape
    yPred = np.dot(X, W)
    mistakes = 0
    for i in range(SIZE):
        if (np.argmax(yPred[i]) != np.argmax(T[i])):
            mistakes += 1
    return ((SIZE - mistakes) / SIZE) * 100

# Calculate loss of model
def calculateLoss(W, DataX, DataT):
    SIZE = DataX.shape[0]  # Define SIZE from number of rows of DataX (Number of Images)
    totalLoss = 0          # Initialize total loss  
    yHat = sp.softmax(np.dot(DataX, W) - np.max(np.dot(DataX, W)), axis=1)  # Perform softmax mapping on computed probabilities for each class
    for i in range(SIZE):  # For loop to iterate through number of images
        yHat[i] = np.interp(yHat[i], [0, 1], [1e-30, .99])     # Perform interpolation on yHat row to eliminate divide by zero error.
        loss = -1.0 * np.log(yHat[i][np.argmax(DataT[i])])  # Calculate loss, neglecting terms with 0 in T matrix since 0 * log(x) = 0 for computational simplicity
        totalLoss += loss                                   # Add loss for current image to total loss
    totalLoss /= float(SIZE)                                # Divide total loss by size of DataX (Number of Images)
    return totalLoss                                        # Return total loss

# Gradient descent with momentum function to fit model
def gradientDescent(DataX, DataT):
    start_time = time.time()  # Define start time
    M = DataX.shape[1]        # Define number of features (M) + 1 (785)
    K = DataT.shape[1]        # Define number of classes (K) (10)
    Beta = 0.85               # Define beta of 0.9
    prevgradient = 0          # Initialize previous gradient to 0
    W = np.random.rand((M),K) # Initialize weight matrix of size (785x10) to random values
    maxEpochs = 5000         # Define max epochs (iterations) to 1000 
    NE = 0                    # Initialize number of iterations to 0
    LR = .001                # Define learning rate of 0.1
    for i in range(maxEpochs):  # For loop to iterate through epochs (iterations)
        temp1 = sp.softmax(np.dot(DataX, W) - np.max(np.dot(DataX, W)), axis=1)             # Define temp1 as softmax mapping of computed probalities with current weight matrix values
        gradient = np.dot(np.transpose(DataX), temp1) - np.dot(np.transpose(DataX), DataT)  # Calculate gradient
        gradient = Beta * prevgradient + (1.0 - Beta) * gradient                            # Calculate gradient with momentum 
        prevW = W
        W = W - LR * gradient                                                               # Calculate new weight matrix
        prevgradient = gradient                                                             # Assign previous gradient the value of current gradient
        NE += 1                                                                             # Increment number of iterations by 1
        loss = calculateLoss(W, DataX, DataT)                                               # Calculate loss
        accuracyPercentage = accuracy(W, DataX, DataT)                                      # Calculate accuracy
        print("Epoch", NE,"- Accuracy (%) {0:.2f}" .format(accuracyPercentage), ", Loss {0:.2f}" .format(loss))  # Print Epochs, Accuracy, and Loss
        DataX, DataT = shuffleData(DataX, DataT)
        if (np.linalg.norm(np.mean(W, axis=0) - np.mean(prevW, axis=0)) < 5e-6):
            print(accuracy(W, DataX, DataT))
            break
    total_time = time.time() - start_time                                                   # Define total time as current time - start time
    return W, NE, total_time                                                                # Return weight matrix, number of iterations, and total time taken


data = np.load('DataXResized.npz')
DataX = data['DataX']
data = np.load('DataTResized.npz')
DataT = data['DataT']

print(DataX.shape)
print(DataT.shape)
DataXTrain = DataX[:math.ceil(DataX.shape[0] * 0.8)]
DataXTest = DataX[math.ceil(DataX.shape[0] * 0.8):]
DataTTrain = DataT[:math.ceil(DataT.shape[0] * 0.8)]
DataTTest = DataT[math.ceil(DataX.shape[0] * 0.8):]
print(DataXTrain.shape)
print(DataXTest.shape)
# W , NE ,TT = gradientDescent(DataXTrain, DataTTrain)

# print("Amount of Time Taken: ", TT)

# np.savez_compressed('W_CElegans.npz', W = W)

trainedModel = np.load('W_CElegans.npz')['W']
print(accuracy(trainedModel, DataXTest, DataTTest))