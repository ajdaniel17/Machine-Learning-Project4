import cv2 as cv
import numpy as np
import idx2numpy

# Load Train Images and Labels Files
imageTrainFile = 'MNIST/DataUncompressed/train-images.idx3-ubyte'
labelTrainFile = 'MNIST/DataUncompressed/train-labels.idx1-ubyte'

# Convert files to numpy arrays
imageTrainArr = idx2numpy.convert_from_file(imageTrainFile)
labelTrainArr = idx2numpy.convert_from_file(labelTrainFile)

# Display using opencv
for i in range(0, imageTrainArr.size):
    cv.imshow(str(labelTrainArr[i]), imageTrainArr[i])
    cv.waitKey()
    cv.destroyAllWindows


