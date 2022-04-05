import numpy as np
import matplotlib.pyplot as plt
import math
import time
import scipy.special as sp

# Function to shuffle data
def shuffleData(DataX, DataT):
    p = np.random.permutation(DataX.shape[0])  # Define number of permutations based on number of images
    return DataX[p], DataT[p]                  # Shuffle DataX and DataT (ensuring image and corresponding label shuffle to the same location)

# Calculate accuracy of model
def calculateAccuracy(W, DataX, DataT):
    SIZE = DataX.shape[0]      # Define SIZE from number of rows of DataX (Number of Images) 
    yHat = np.dot(DataX, W)    # Compute probabilities for each class
    errors = 0                 # Initialize errors to 0
    for i in range(SIZE):      # For loop to iterate through number of images
        if (np.argmax(yHat[i]) != np.argmax(DataT[i])):  # If index of max computed probability is not equal to index of max actual probability from one-hot encoding of labels 
            errors += 1                                  # Increment error by 1
    return ((SIZE - errors) / SIZE) * 100                # Compute and return accuracy as a percentage

# Calculate loss of model
def calculateLoss(W, DataX, DataT):
    SIZE = DataX.shape[0]  # Define SIZE from number of rows of DataX (Number of Images)
    totalLoss = 0          # Initialize total loss  
    yHat = sp.softmax(np.dot(DataX, W) - np.max(np.dot(DataX, W)), axis=1)  # Perform softmax mapping on computed probabilities for each class
    for i in range(SIZE):  # For loop to iterate through number of images
        yHat[i] = np.interp(yHat[i], [0, 1], [.1, .99])     # Perform interpolation on yHat row to eliminate divide by zero error.
        loss = -1.0 * np.log(yHat[i][np.argmax(DataT[i])])  # Calculate loss, neglecting terms with 0 in T matrix since 0 * log(x) = 0 for computational simplicity
        totalLoss += loss                                   # Add loss for current image to total loss
    totalLoss /= float(SIZE)                                # Divide total loss by size of DataX (Number of Images)
    return totalLoss                                        # Return total loss

# Gradient descent with momentum function to fit model
def gradientDescent(DataX, DataT):
    start_time = time.time()    # Define start time
    M = DataX.shape[1]          # Define number of features (M) + 1 (785)
    K = DataT.shape[1]          # Define number of classes (K) (2)
    Beta = 0.85                 # Define beta of 0.85
    prevgradient = 0            # Initialize previous gradient to 0
    W = np.random.rand((M),K)   # Initialize weight matrix of size (785x2) to random values
    maxEpochs = 5000            # Define max epochs (iterations) to 5000 
    NE = 0                      # Initialize number of iterations to 0
    LR = 0.001                  # Define learning rate of 0.001
    for i in range(maxEpochs):  # For loop to iterate through epochs (iterations)
        temp1 = sp.softmax(np.dot(DataX, W) - np.max(np.dot(DataX, W)), axis=1)             # Define temp1 as softmax mapping of computed probalities with current weight matrix values
        gradient = np.dot(np.transpose(DataX), temp1) - np.dot(np.transpose(DataX), DataT)  # Calculate gradient
        gradient = Beta * prevgradient + (1.0 - Beta) * gradient                            # Calculate gradient with momentum 
        prevW = W                                                                           # Assign prevW value of current W
        W = W - LR * gradient                                                               # Calculate new weight matrix
        prevgradient = gradient                                                             # Assign previous gradient the value of current gradient
        NE += 1                                                                             # Increment number of iterations by 1
        DataX, DataT = shuffleData(DataX, DataT)                                            # Shuffle Data
        if (np.linalg.norm(np.mean(W, axis=0) - np.mean(prevW, axis=0)) < 1e-7):            # If weight vector is not changing greatly,
            break                                                                           # break loop
    total_time = time.time() - start_time                                                   # Define total time as current time - start time
    return W, NE, total_time                                                                # Return weight matrix, number of iterations, and total time taken


DataX = np.load('DataXProcessed.npz')['DataX']  # Load preprocessed numpy matrix for images
DataT = np.load('DataTProcessed.npz')['DataT']  # Load preprocessed numpy matrix for labels

# Divide DataX into training and testing sets
DataXTrain = DataX[:math.ceil(DataX.shape[0] * 0.8)]
DataXTest = DataX[math.ceil(DataX.shape[0] * 0.8):]

# Divide DataT into training and testing sets
DataTTrain = DataT[:math.ceil(DataT.shape[0] * 0.8)]
DataTTest = DataT[math.ceil(DataX.shape[0] * 0.8):]

W , NE ,TT = gradientDescent(DataXTrain, DataTTrain)

# np.savez_compressed('W_CElegans.npz', W = W)                               # Save weight vector as compressed numpy array
# trainedModel = np.load('W_CElegans.npz')['W']                              # Load saved weight vector
print("Number of Iterations - ", NE, ", Total Time (s)", TT)                 # Print number of iterations and total time taken
trainLoss = calculateLoss(W, DataXTrain, DataTTrain)                         # Calculate training loss
trainAccuracy = calculateAccuracy(W, DataXTrain, DataTTrain)                 # Calculate training accuracy
print('Training - Loss =', trainLoss, ", Accuracy (%) = ", trainAccuracy)    # Print training loss and accuracy
testLoss = calculateLoss(W, DataXTest, DataTTest)                            # Calculate testing loss
testAccuracy = calculateAccuracy(W, DataXTest, DataTTest)                    # Calculate testing accuracy
print('Testing - Loss =', testLoss, ", Accuracy (%) = ", testAccuracy)            # Print testing loss and accuracy