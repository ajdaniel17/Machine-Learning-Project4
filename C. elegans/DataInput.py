#NOTE: This takes a while, this program takes both datasets in the folders 0 and 1 and linerizes the data (in greyscale) 

import numpy as np
import os
from PIL import Image

SIZE = 101*101 #Size of the images 

#Load all images in Data set 0 into a list
print("LOADING WORM DATA SET 0")
imgs = []
path = os.getcwd() + "/C. elegans/Data/0"
valid_images = [".png"]
for f in os.listdir(path):
    ext = os.path.splitext(f)[1]
    if ext.lower() not in valid_images:
        continue
    imgs.append(Image.open(os.path.join(path,f)))

#Take all images loaded into list and convert them into greyscale, reshape them to be 1 x SIZE, then append them to numpy array NoWorm
print("FORMATTING LIST INTO NUMPY ARRAY")
NoWorm = np.empty((0,SIZE),int)
for i in range(len(imgs)):
    temp = np.reshape(imgs[i].convert('L'),SIZE)
    temp = temp.astype(int)
    temp = np.array([temp])
    NoWorm  = np.append(NoWorm,temp,0)
print("Shape of Formatted Numpy Array: ", NoWorm.shape)

#Load all images in Data set 1 into a list
print("LOADING WORM DATA SET 1")
imgs = []
path = os.getcwd() + "/C. elegans/Data/1"
for f in os.listdir(path):
    ext = os.path.splitext(f)[1]
    if ext.lower() not in valid_images:
        continue
    imgs.append(Image.open(os.path.join(path,f)))

#Take all images loaded into list and convert them into greyscale, reshape them to be 1 x SIZE, then append them to numpy array Worm
print("FORMATTING LIST INTO NUMPY ARRAY")
Worm = np.empty((0,SIZE),int)
for i in range(len(imgs)):
    temp = np.reshape(imgs[i].convert('L'),SIZE)
    temp = temp.astype(int)
    temp = np.array([temp])
    Worm  = np.append(Worm,temp,0)
print("Shape of Formatted Numpy Array: ",Worm.shape)