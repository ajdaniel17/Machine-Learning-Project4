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
y =labelTrainArr #training labels

#one-hot encoding the labels
def one_hot(y, c):
    y_hot = np.zeros((len(y), c))
    y_hot[np.arange(len(y)), y] = 1
    return y_hot

def softmax(z):
    # z --> linear
    exp = np.exp(z-np.max(z))

    for i in range(len(z)):
        exp[i] /= np.sum(exp[i])

    return exp

#training 

def model(X, label, rho, c, iter):
    #X_train = MNIST training input
    # y = labels (true model)
    # rho = learning rate
    #c = number of classes
    #iter = iterations

    #m = number of training examples
    #n = number of features (i.e. linearized pixels)
    m, n = X.shape
    print(m)
    print(n)
    print(c)

    #initialize weight vector to a random value and bias
    w = np.zeros((n, c))
    bias = np.random.random(c)

    #create empty array to store loss for each iteration
    loss_values = []

    for i in range(iter):

        #model prediction and softmax mapping
        z = X_train@w + bias
        y_hat = softmax(z)
        print('1',y_hat.shape)
        #one-hot encoding labels
        y_hot = one_hot(label,c)
        print('2', y_hot.shape)
        #gradient descent and bias calculation
        w_gradient = (1/m)*np.dot(X_train.T, (y_hat-y_hot))
        print("3", w_gradient.shape)
        bias_gradient = (1/m) *np.sum(y_hat-y_hot)

        #weight vector and bias update
        w = w - rho*w_gradient
        bias = bias - rho*bias_gradient

        #calculating loss function for the given iteration
        loss = -np.mean(np.log(y_hat[np.arange(len(y)), y]))
        loss_values.append(loss)

        #printing loss value at every 100th iteration to examine accuracy of model
        if i%100==0:
            print('Iteration {i}==> Loss = {loss}'.format(i = i, loss=loss))

    return w, bias, loss_values 

c = 10
rho = 0.50
iter = 1000

w, bias, l = model(X_train, y, rho, c, iter)

#now, using model to make predictions of the classifiers

def predict(X , w, b):
    #X --> input
    #w --> weight vector
    #b --> bias

    z = X_train@w + bias
    y_hat = softmax(z)

    return np.argmax(y_hat, axis=1)

def accuracy(y, y_hat):
    return np.sum(y==y_hat)/len(y)

train_labels = predict(X_train, w, bias)
training_accuracy = accuracy(y, train_labels)

print(training_accuracy)




