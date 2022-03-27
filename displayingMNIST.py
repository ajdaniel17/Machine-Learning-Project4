import cv2 as cv
import numpy as np
import idx2numpy
import matplotlib.pyplot as plt

# Load Train Images and Labels Files
imageTrainFile = 'MNIST/DataUncompressed/train-images.idx3-ubyte'
labelTrainFile = 'MNIST/DataUncompressed/train-labels.idx1-ubyte'

# Convert files to numpy arrays
imageTrainArr = idx2numpy.convert_from_file(imageTrainFile)
labelTrainArr = idx2numpy.convert_from_file(labelTrainFile)

fig = plt.figure(figsize=(12, 9))

for i in range(30):
    ax = fig.add_subplot(5, 6, i+1)
    ax.imshow(imageTrainArr[i], cmap=plt.get_cmap('gray'))
    ax.set_title('Label: {y}' .format(y=labelTrainArr[i]))
    plt.axis('off')
plt.show()

# Display using opencv
# for i in range(0, imageTrainArr.size):
#     cv.imshow(str(labelTrainArr[i]), imageTrainArr[i])
#     cv.waitKey()
#     cv.destroyAllWindows


