import time
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

# Calculate loss of model
def calculateLoss(W, DataX, DataT):
    SIZE = DataX.shape[0]  # Define SIZE from number of rows of DataX (Number of Images)
    totalLoss = 0          # Initialize total loss  
    yHat = sp.softmax(np.dot(DataX, W) - np.max(np.dot(DataX, W)), axis=1)  # Perform softmax mapping on computed probabilities for each class
    for i in range(SIZE):  # For loop to iterate through number of images
        yHat[i] = np.interp(yHat[i], [0, 1], [1e-15, .99])     # Perform interpolation on yHat row to eliminate divide by zero error.
        loss = -1.0 * np.log(yHat[i][np.argmax(DataT[i])])  # Calculate loss, neglecting terms with 0 in T matrix since 0 * log(x) = 0 for computational simplicity
        totalLoss += loss                                   # Add loss for current image to total loss
    totalLoss /= float(SIZE)                                # Divide total loss by size of DataX (Number of Images)
    return totalLoss                                        # Return total loss

# Calculate accuracy of model
def accuracy(W, DataX, DataT):
    SIZE = DataX.shape[0]      # Define SIZE from number of rows of DataX (Number of Images) 
    yHat = np.dot(DataX, W)    # Compute probabilities for each class
    errors = 0                 # Initialize errors to 0
    for i in range(SIZE):      # For loop to iterate through number of images
        if (np.argmax(yHat[i]) != np.argmax(DataT[i])):  # If index of max computed probability is not equal to index of max actual probability from one-hot encoding of labels 
            errors += 1                                  # Increment error by 1
    return ((SIZE - errors) / SIZE) * 100                # Compute and return accuracy as a percentage

# Generate T array from labels array (One-Hot Encoding)
def generateT(N, K, labelArr):
    DataT = np.zeros((N, K))  # Initialize DataT matrix to zeros of Size (NxK) 
    for i in range(0, labelArr.shape[0]):  # For loop to iterate through all labels
        DataT[i, labelArr[i]] = 1          # Create one-hot encoded row
    return DataT                           # Return one-hot encoded matrix

# Normalize pixel values and generate data matrix
def normalizeAndGenerateDataMatrix(imageArr):
    imageArr = imageArr / 255  # Normalize 
    return np.insert(imageArr, imageArr.shape[1], 1, axis=1)

# Gradient descent with momentum function to fit model
def gradientDescent(DataX, DataT, LabelArr):
    start_time = time.time()
    SIZE, D = DataX.shape
    SIZE, K = DataT.shape
    Beta = .9
    prevgradient = 0
    W = np.random.rand((D),K)
    maxEpochs = 1000
    NE = 0
    LR = .01
    for i in range(maxEpochs):
        temp1 = sp.softmax(np.dot(DataX, W) - np.max(np.dot(DataX, W)), axis=1)
        gradient = np.dot(np.transpose(DataX), temp1) - np.dot(np.transpose(DataX), DataT)
        gradient = Beta * prevgradient + (1.0 - Beta) * gradient
        W = W - LR * gradient
        prevgradient = gradient
        NE += 1
        loss = calculateLoss(W, DataX, DataT)
        accuracyPercentage = accuracy(W, DataX, DataT)
        print("Epoch", NE,"- Accuracy (%) {0:.2f}" .format(accuracyPercentage), ", Loss {0:.2f}" .format(loss))
    total_time = time.time() - start_time 
    return W, NE, total_time

# Number of Classes
K = 10
# Number of Images
N = imageTrainArrLinearized.shape[0]
# Number of Features
M = imageTrainArrLinearized.shape[1]
T = generateT(N, K, labelTrainArr)
imageDataMatrixTrain = normalizeAndGenerateDataMatrix(imageTrainArrLinearized)
trainedModel, numIter, totalTime = gradientDescent(imageDataMatrixTrain, T, labelTrainArr)