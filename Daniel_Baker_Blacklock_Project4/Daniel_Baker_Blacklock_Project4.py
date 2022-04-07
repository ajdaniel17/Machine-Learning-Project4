import numpy as np
import glob
import cv2 as cv

path = input("Please enter the directory path containing test images:\n")

filenamesCElegans = glob.glob(path + '/*.png')
filenamesCElegans.sort()
filenamesMNIST = glob.glob(path + '/*.tif')
filenamesMNIST.sort()

def resizeAndConvertToGrayscale(filenames):
    images = [cv.imread(img) for img in filenames]

    scale_percent = 28
    width = int(101 * scale_percent / 100)
    height = int(101 * scale_percent / 100)
    dim = (width, height)

    imagesResized = [cv.resize(img, dim, interpolation=cv.INTER_AREA) for img in images]
    imagesGrayscale = [cv.cvtColor(img, cv.COLOR_BGR2GRAY) for img in imagesResized]

    return imagesGrayscale

def printResults(filenames, W, yPred):
    print(" ___________________________________ ")
    print("|                    |              |")
    print("|    Image Name      |     Class    |")
    print("|                    |              |")
    print(" ___________________________________ ")

    K = W.shape[1]

    totals = np.zeros((K))
    for i in range(len(filenames)):
        print("|  %s   |     %i      |" % (filenames[i].replace(path+"\\",""),np.argmax(yPred[i])))
        totals[np.argmax(yPred[i])] += 1

    print("Total Tallies:")
    for i in range(K):
        print("Class", int(i) , ":" , int(totals[i]))

if filenamesCElegans:
    imagesGrayscale = resizeAndConvertToGrayscale(filenamesCElegans)
    imagesBlurred = [cv.GaussianBlur(img, (3,3), 0) for img in imagesGrayscale]

    sobelx = [cv.Sobel(src=img, ddepth=cv.CV_64F, dx=1, dy=0, ksize=5) for img in imagesBlurred]
    sobely = [cv.Sobel(src=img, ddepth=cv.CV_64F, dx=0, dy=1, ksize=5) for img in imagesBlurred]
    sobelxy = [cv.Sobel(src=img, ddepth=cv.CV_64F, dx=1, dy=1, ksize=5) for img in imagesBlurred]

    edges = np.array([cv.Canny(image=img, threshold1=60, threshold2=140) for img in imagesBlurred])

    imagesLinearized = edges.reshape(edges.shape[0], edges.shape[1] * edges.shape[2])

    DataX = imagesLinearized / 255
    DataX =  np.insert(DataX, DataX.shape[1], 1, axis=1)

    W = np.load('W_CElegans.npz')['W']

    yPred = np.dot(DataX, W)

    printResults(filenamesCElegans, W, yPred)
    

if filenamesMNIST:
    imagesGrayscale = np.array(resizeAndConvertToGrayscale(filenamesMNIST))

    imagesLinearized = imagesGrayscale.reshape(imagesGrayscale.shape[0], imagesGrayscale.shape[1] * imagesGrayscale.shape[2])

    DataX = imagesLinearized / 255
    DataX =  np.insert(DataX, DataX.shape[1], 1, axis=1)

    W = np.load('trainedModelMNIST.npz')['x']

    yPred = np.dot(DataX, W)

    printResults(filenamesMNIST, W, yPred)
