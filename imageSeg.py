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

    filtflip = np.flip(filter)

    current_y = 0
    out_y = 0
    while current_y + filty <= imgy:
        current_x = 0
        out_x = 0
        while current_x + filtx <= imgx:
            a = filtflip * img[current_y:current_y + filty, current_x:current_x + filtx]
            outmtx[out_y, out_x] = a.sum() + bias
            current_x += 1
            out_x += 1
        current_y += 1
        out_y += 1

    cache = (img, filtflip)  # Storing for later backpropagation

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
    try:
        return math.exp(inp)/(math.exp(inp) + 1)
    except OverflowError:
        return math.exp(99)/(math.exp(99) + 1)


def deriveact(act):             # input activation result, output the derivative
    return act*(1-act)

def run(o, d, w1, w2):
    c = convolution(o, d)[0]
    z = maxpool(c)  # z[0] is max pool, z[1] is location of maxes
    i = z[0].flatten()
    n = np.zeros((8649, 1))  # list of input nodes after activation

    for x in range(i.size):
        n[x] = (round(activation(i[x]), 2))

    feedforward = np.matmul(w1, n)  # Feed input through first layer
    for i in range(feedforward.size):
        feedforward[i] = activation(feedforward[i])

    feedforward = np.matmul(w2, feedforward)  # Feed input through second layer
    for i in range(feedforward.size):
        feedforward[i] = activation(feedforward[i])

    error = np.zeros((2, 1))
    error[0] = feedforward[0] - 1
    error[1] = feedforward[1]

    e = (feedforward[0] - 1) ** 2 + (feedforward[1]) ** 2
    print("Error =", e, " ***************************")
    print(feedforward)
    if feedforward[0] > 0.70:
        print("It's a car!")
    else:
        print("It is not a car")


def train(o, d, w1, w2, learnfactor):
    c = convolution(o, d)[0]
    z = maxpool(c)                                    # z[0] is max pool, z[1] is location of maxes
    i = z[0].flatten()
    n = np.zeros((8649, 1))                              # list of input nodes after activation

    for x in range(i.size):
        n[x] = (round(activation(i[x]), 2))

    # print(n, "are the inputs to the net\n")            # Prints inputs into the dense network

    feedforward = np.matmul(w1, n)                       # Feed input through first layer
    for i in range(feedforward.size):
        feedforward[i] = activation(feedforward[i])

    actlayer1 = feedforward.copy()                       # Saves the output after activation for layer 1

    feedforward = np.matmul(w2, feedforward)             # Feed input through second layer
    for i in range(feedforward.size):
        feedforward[i] = activation(feedforward[i])

    deltasave = feedforward.copy()

    error = np.zeros((2, 1))
    error[0] = feedforward[0] - 1
    error[1] = feedforward[1]

    e = (feedforward[0] - 1)**2 + (feedforward[1])**2
    print("Error =", e, " ***************************")
    print(deltasave)

    errsig2 = np.zeros((2, 1))                                   # Begin calculating new weights for w2
    for i in range(errsig2.size):
        errsig2[i][0] = error[i] * deriveact(deltasave[i])

    weightgrad2 = np.matmul(errsig2, np.transpose(actlayer1))

    w2 = np.subtract(w2, weightgrad2)

    errsig1 = np.zeros((1000, 1))                                # Begin calculating new weights for w1

    for i in range(errsig1.size):
        errsig1[i][0] = error[0] * deriveact(deltasave[0]) + error[1] * deriveact(deltasave[1])

    # weightgrad = np.add(errsig1, errsig1_2)

    weightgrad = np.matmul(errsig1, np.transpose(n))
    # print(weightgrad)

    w1 = np.subtract(w1, weightgrad)

    # feedforward[0] = feedforward[0] - 1

    # print(delta, "is Delta\n")

    wt = w1.transpose()
    wt = np.matmul(wt, errsig1)

    gradientmapin = np.zeros((8649, 1))

    for i in range(n.size):
        gradientmapin[i] = wt[i] * deriveact(n[i])

    # print(gradientmapin, "\n")

    revmp = np.zeros(c.shape)

    for i in range(gradientmapin.size):
        revmp[z[1][i][1]][z[1][i][0]] = gradientmapin[i]        # puts the gradients in place to reverse max pooling

    # print(revmp)

    congradient = learnfactor * np.around(convolution(o, revmp)[0], 2)

    # print(congradient)

    d = np.subtract(d, congradient)                             # New Kernel
    # print(d, "New Kernel\n")

    return d, w1, w2


imageLocation = '\\Columbus_CSUAV_AFRL\\train'        # Where training images are stored relative to the CWD
carpics = []
o = []
for i in range(11):
    carpics.append(getFileImages(os.getcwd() + imageLocation)[0][i])
    img = Image.open(carpics[i]).convert('L')
    o.append(np.asarray(img))

negpics = []
n = []
for i in range(11):
    negpics.append(getFileImages(os.getcwd() + imageLocation)[0][i])
    img = Image.open(negpics[i]).convert('L')
    n.append(np.asarray(img))

stddev = 1/np.sqrt(np.prod(4900))
d = []
for i in range(3):
    d.append(np.random.normal(loc=0, scale=stddev, size=4900))  # Initializes kernel randomly on a normal distribution
    d[i] = np.reshape(d[i], (70, 70))


'''
stddev = 1/np.sqrt(np.prod(4))
d = np.random.normal(loc=0, scale=stddev, size=4)  # Initializes kernel randomly on a normal distribution
d = np.reshape(d, (2, 2))
d = np.around(d, 2)'''

stddev = 1/np.sqrt(np.prod(2000))
w2 = np.random.normal(loc=0, scale=stddev, size=2000)  # Initializes weights randomly on a normal distribution
w2 = np.reshape(w2, (2, 1000))                             # Reshape to a 2x1000 matrix for multiplication
w2 = np.around(w2, 2)
print(w2, " are the initial weights for w2\n")             # Prints hidden layer of weights

stddev = 1/np.sqrt(np.prod(8649000))
w1 = np.random.normal(loc=0, scale=stddev, size=8649000)  # Initializes weights randomly on a normal distribution
w1 = np.reshape(w1, (1000, 8649))                          # Reshape to a 1000x8649 matrix for multiplication
w1 = np.around(w1, 2)
print(w1, " are the initial weights for w1\n")             # Prints hidden layer of weights


learnfactor = 10

for i in range(1):
    for k in range(10):
        for j in range(len(d)):                                      # Train all the filters
            d[j], w1, w2 = train(o[k], d[j], w1, w2, learnfactor)

run(o[10], d[1], w1, w2)

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

o = np.array([(0.51, 0.9, 0.88, 0.84, 0.05),
              (0.4, 0.62, 0.22, 0.59, 0.1),
              (0.11, 0.2, 0.74, 0.33, 0.14),
              (0.47, 0.01, 0.85, 0.7, 0.09),
              (0.76, 0.19, 0.72, 0.17, 0.57)])

'''




