import numpy as np
import glob
import cv2 as cv

#INSERT PATH HERE
path = "C:/Users/ajdan/Documents/PR-ML/Machine Learning/Project 4 ML/Machine-Learning-Project4/C. elegans/Data/1"

data = np.load('W_CElegans.npz')

W = data['W']
SIZE , K = W.shape

SIZE = 28*28
filenames = glob.glob(path + '/*.png')
filenames.sort()
images = [cv.imread(img) for img in filenames]

scale_percent = 28 # percent of original size
width = int(101 * scale_percent / 100)
height = int(101 * scale_percent / 100)
dim = (width, height)

imagesResized = [cv.resize(img, dim, interpolation=cv.INTER_AREA) for img in images]
imagesGrayscale = [cv.cvtColor(img, cv.COLOR_BGR2GRAY) for img in imagesResized]
imagesBlurred = [cv.GaussianBlur(img, (3,3), 0) for img in imagesGrayscale]

sobelx = [cv.Sobel(src=img, ddepth=cv.CV_64F, dx=1, dy=0, ksize=5) for img in imagesBlurred]
sobely = [cv.Sobel(src=img, ddepth=cv.CV_64F, dx=0, dy=1, ksize=5) for img in imagesBlurred]
sobelxy = [cv.Sobel(src=img, ddepth=cv.CV_64F, dx=1, dy=1, ksize=5) for img in imagesBlurred]

edges = [cv.Canny(image=img, threshold1=60, threshold2=140) for img in imagesBlurred]


DataX = np.empty((0,SIZE),int)
for i in range(len(edges)):
    temp = np.reshape(edges[i],SIZE)
    temp = temp.astype(int)
    temp = np.array([temp])
    DataX  = np.append(DataX,temp,0)

DataX = DataX / 255
DataX =  np.insert(DataX, DataX.shape[1], 1, axis=1)

print(W.shape)
print(DataX.shape)

yPred = np.dot(DataX, W)

print(" ___________________________________ ")
print("|                    |              |")
print("|    Image Name      |     Class    |")
print("|                    |              |")
print(" ___________________________________ ")

totals = np.zeros((K))
for i in range(len(filenames)):
    print("|  %s   |     %i      |" % (filenames[i].replace(path+"\\",""),np.argmax(yPred[i])))
    totals[np.argmax(yPred[i])] += 1

print("Total Tallies:")
for i in range(K):
    print("Class", int(i) , ":" , int(totals[i]))
