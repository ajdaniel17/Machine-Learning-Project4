#implementation of logistic regression?
#import libraries and dataset first
import cv2 as cv
import numpy as np
import idx2numpy
import matplotlib.pyplot as plt
import scipy
from scipy.special import softmax



# Load Train Images and Labels Files
imageTrainFile = 'MNIST/DataUncompressed/train-images.idx3-ubyte'
labelTrainFile = 'MNIST/DataUncompressed/train-labels.idx1-ubyte'

#Load test image and test labels
imageTestFile = 't10k-images-idx3-ubyte.gz'



# Convert files to numpy arrays
imageTrainArr = idx2numpy.convert_from_file(imageTrainFile)
labelTrainArr = idx2numpy.convert_from_file(labelTrainFile)


#visualize one 28x28 matrix of data
#print(imageTrainArr[0])
#print(labelTrainArr[0])

#28x28 = 784, training set size of 60000 --> resize training set to (60000, 784) input matrix

X_train = imageTrainArr.reshape(60000, 784)

#noramlizing training data

X_train = X_train / 255
ones = np.ones(60000)
phi_train = np.column_stack([X_train,ones.transpose()])

#now, want to encode y values (i.e. classifiers)
t_train =labelTrainArr #training labels

#implementation of gradient descent, softmax mapping, minimization of loss function, etc.

#first, softmax function def
y_train = np.zeros((60000, 10))

#acquiring dimensions of X_train
m, n = X_train.shape
#initializing loss function
J = 0
J_total = 0

#initializing B and rho parameters for GD w/momentum
B = 0.9
rho = 0.25

#number of iterations
iter = 0

#randomized initial "guess" for w
w = np.random.random((n + 1, 10))

#softmax mapping of training data using initial guess for w
y_train = softmax(phi_train@w)

#calculation of the loss value for this approximation


#pulling classifiers from softmax mapping output
y_hat = np.argmax(y_train, axis=1)

#calculating loss
for i in range (10):
    J -= t_train*np.log10(y_train[:,i])

for k in range(60000): 
    J_total = np.sum(J[k])

loss = J_total / 60000


for i in range(60000):
    grad = np.sum((y_hat[i]-t_train[i])*phi_train[i,:])

print(loss)

