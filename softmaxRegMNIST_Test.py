import numpy as np
import idx2numpy
import matplotlib.pyplot as plt
import scipy.special as sp

# Calculate accuracy of model
def accuracy(W, DataX, DataT):
    SIZE = DataX.shape[0]      # Define SIZE from number of rows of DataX (Number of Images) 
    yHat = sp.softmax(np.dot(DataX, W) - np.max(np.dot(DataX, W)), axis=1)    # Compute probabilities for each class
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

imageTestFile = 'MNIST/DataUncompressed/test-images.idx3-ubyte'  # Load Test Images File
labelTestFile = 'MNIST/DataUncompressed/test-labels.idx1-ubyte'  # Load Test Labels File

imageTestArr = idx2numpy.convert_from_file(imageTestFile)  # Convert Testing images to numpy arrays
labelTestArr = idx2numpy.convert_from_file(labelTestFile)  # Convert Testing labels to numpy arrays

imageTestArrLinearized = imageTestArr.reshape(imageTestArr.shape[0], imageTestArr.shape[1] * imageTestArr.shape[2])  # Linearize images

trainedModel = np.load('trainedModelMNIST.npz')['x']

K = 10                                  # Number of Classes
N = imageTestArrLinearized.shape[0]     # Number of Images
M = imageTestArrLinearized.shape[1]     # Number of Features
dataMatrixTest = normalizeAndGenerateDataMatrix(imageTestArrLinearized)
TTest = generateT(N, K, labelTestArr)
print(accuracy(trainedModel, dataMatrixTest, TTest))

