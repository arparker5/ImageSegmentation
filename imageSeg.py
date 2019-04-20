from PIL import Image
from skimage import data
from matplotlib import pyplot as plt
import numpy as np
import scipy as sp
import sys
import glob
import os
import math


def getFileImages(cwd):                 # Extracts training images of cars and negatives and puts them into two lists
    print("Extracting test images...")
    carimagecount = 0
    negimagecount = 0
    carimage_list = []
    negimage_list = []
    st = os.path.join(cwd, "*.png")

    for imagename in glob.glob(st):
        if imagename[89] == 'c':
            carimage_list.append(imagename)
            carimagecount += 1
        else:
            negimage_list.append(imagename)
            negimagecount += 1
    print("Extracted ", carimagecount, " car images")
    print("Extracted ", negimagecount, " neg images")
    return carimage_list, negimage_list


def convolution(img, filter, bias=0, s=1):
    (imgy, imgx) = img.shape
    (filty, filtx) = filter.shape

    outdim_x = int((imgx - filtx)/s) + 1
    outdim_y = int((imgy - filty)/s) + 1
    outmtx = np.zeros((outdim_y, outdim_x))

    current_y = 0
    out_y = 0
    while current_y + filty <= imgy:
        current_x = 0
        out_x = 0
        while current_x + filtx <= imgx:
            a = filter * img[current_y:current_y + filty, current_x:current_x + filtx]
            outmtx[out_y, out_x] = a.sum() + bias
            current_x += 1
            out_x += 1
        current_y += 1
        out_y += 1

    cache = (img, filter)  # Storing for later backpropagation

    return outmtx, cache


def maxpool(img, filt=2, s=2):
    (imgy, imgx) = img.shape

    outdim_x = int((imgx - filt)/s) + 1
    outdim_y = int((imgy - filt)/s) + 1

    outmtx = np.zeros((outdim_y, outdim_x))
    maxpos = []                                 # Holds the indices of the max

    current_y = 0
    out_y = 0
    while current_y + filt <= imgy:
        current_x = 0
        out_x = 0
        while current_x + filt <= imgx:
            a = img[current_y:current_y + filt, current_x:current_x + filt]
            outmtx[out_y, out_x] = np.max(a)
            ind = np.unravel_index(np.argmax(a, axis=None), a.shape)
            maxpos.append((ind[1] + current_x, ind[0] + current_y))
            current_x += s
            out_x += 1
        current_y += s
        out_y += 1

    return outmtx, maxpos


def activation(inp):                             # Sigmoid function
    return math.exp(inp)/(math.exp(inp) + 1)


def deriveact(act):             # input activation result, output the derivative
    return act*(1-act)


o = np.array([(0.51, 0.9, 0.88, 0.84, 0.05),
              (0.4, 0.62, 0.22, 0.59, 0.1),
              (0.11, 0.2, 0.74, 0.33, 0.14),
              (0.47, 0.01, 0.85, 0.7, 0.09),
              (0.76, 0.19, 0.72, 0.17, 0.57)])

d = np.array([(-0.13, 0.15),
              (-0.51, 0.62)])

c = convolution(o, d)[0]
z = maxpool(c)                                    # z[0] is max pool, z[1] is location of maxes
i = z[0].flatten()
n = np.zeros((4, 1))                              # list of nodes after activation

for x in range(i.size):
    n[x] = (round(activation(i[x]), 2))

print(n, "are the inputs to the net\n")            # Prints inputs into the dense network

stddev = 1/np.sqrt(np.prod(8))
w = np.random.normal(loc=0, scale=stddev, size=8)  # Initializes weights randomly on a normal distribution
w = np.reshape(w, (2, 4))                          # Reshape to a 2x4 matrix for multiplication
w = np.around(w, 2)
print(w, " are the weights\n")                                     # Prints hidden layer of weights

delta = np.matmul(w, n)
print(delta, "\n\n ----- Begin Backpropagation -----\n")   # Network output
                                                           # begin backpropagation
delta[0] = delta[0] - 1

print(delta, "is Delta\n")

wt = w.transpose()
wt = np.matmul(wt, delta)

gradientmapin = np.zeros((4, 1))

for i in range(n.size):
    gradientmapin[i] = wt[i] * deriveact(n[i])

print(gradientmapin)

revmp = np.zeros(c.shape)

for i in range(gradientmapin.size):
    revmp[z[1][i][1]][z[1][i][0]] = gradientmapin[i]

print(revmp)



'''
np.set_printoptions(threshold=sys.maxsize)
img = Image.open("pixilimg.png").convert('L')
img.save('imggrey.png')
imgg = np.asarray(img)

filt = Image.open("filter.png").convert('L')
filt.save('filtergrey.png')
filtg = np.asarray(filt)


conimg = convolution(imgg, filtg)[0]

mp = maxpool(conimg)

print(conimg.shape)
print(mp.shape)

imageLocation = '\\Columbus_CSUAV_AFRL\\train'        # Where training images are stored relative to the CWD
getFileImages(os.getcwd() + imageLocation)
'''




