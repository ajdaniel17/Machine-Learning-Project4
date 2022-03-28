#implementation of logistic regression?
#import libraries and dataset first
import cv2 as cv
import numpy as np
import idx2numpy
import matplotlib.pyplot as plt

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

#now, want to encode y values (i.e. classifiers)
t_train =labelTrainArr #training labels

#implementation of gradient descent, softmax mapping, minimization of loss function, etc.

#first, softmax function def
y_train = np.zeros((60000, 10))

#initial "guess" for w
w = np.random.random((784,10))

def softmax(w, X_train):

    for j in range(10):
        den = np.zeros(10)
        num = np.exp(np.dot(w[: , j], X_train.transpose()))
        for k in range(10):
            den[k] = np.exp(np.dot(w[:,k],X_train.transpose()))
            total_den = total_den + den[k]
        y_train[:,j] = num / total_den
        return y_train

softmax_out = softmax(w,X_train)
print(softmax_out.ndim)
