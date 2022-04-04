import time
import numpy as np
import idx2numpy
import matplotlib.pyplot as plt
import scipy.special as sp

imageTrainFile = 'MNIST/DataUncompressed/train-images.idx3-ubyte'  # Load Train Images File
labelTrainFile = 'MNIST/DataUncompressed/train-labels.idx1-ubyte'  # Load Train Labels File

imageTrainArr = idx2numpy.convert_from_file(imageTrainFile)  # Convert training images to numpy arrays
labelTrainArr = idx2numpy.convert_from_file(labelTrainFile)  # Convert training labels to numpy arrays

imageTrainArrLinearized = imageTrainArr.reshape(imageTrainArr.shape[0], imageTrainArr.shape[1] * imageTrainArr.shape[2])  # Linearize images

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
    imageArr = imageArr / 255  # Normalize pixels of images to range between 0 - 1
    return np.insert(imageArr, imageArr.shape[1], 1, axis=1)  # Insert Column of 1's at end of each image and Return Matrix

# Gradient descent with momentum function to fit model
def gradientDescent(DataX, DataT):
    start_time = time.time()    # Define start time
    M = DataX.shape[1]          # Define number of features (M) + 1 (785)
    K = DataT.shape[1]          # Define number of classes (K) (10)
    Beta = 0.9                  # Define beta of 0.9
    prevgradient = 0            # Initialize previous gradient to 0
    W = np.random.rand((M),K)   # Initialize weight matrix of size (785x10) to random values
    maxEpochs = 1000            # Define max epochs (iterations) to 1000 
    NE = 0                      # Initialize number of iterations to 0
    LR = .01                    # Define learning rate of 0.1
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
        # print("Epoch", NE,"- Accuracy (%) {0:.2f}" .format(accuracyPercentage), ", Loss {0:.2f}" .format(loss))  # Print Epochs, Accuracy, and Loss
        if (np.linalg.norm(W - prevW) < 1e-3):
            break
    total_time = time.time() - start_time                                                   # Define total time as current time - start time
    return W, NE, total_time                                                                # Return weight matrix, number of iterations, and total time taken

K = 10                                # Number of Classes
N = imageTrainArrLinearized.shape[0]  # Number of Images
M = imageTrainArrLinearized.shape[1]  # Number of Features
TTrain = generateT(N, K, labelTrainArr)    # Generate T Matrix from labels (one-hot encoding)
imageDataMatrixTrain = normalizeAndGenerateDataMatrix(imageTrainArrLinearized)              # Normalize and generate data matrix from images array
trainedModel, numIter, totalTime = gradientDescent(imageDataMatrixTrain, TTrain)  # Fit model 
print("Number of Iterations", numIter, "Total Time", totalTime)