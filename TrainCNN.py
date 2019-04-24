from PIL import Image
from skimage import data
from matplotlib import pyplot as plt
import numpy as np
import scipy as sp
import sys
import glob
import os
import math


def scale(X, x_min, x_max):              # Normalizes the input image
    nom = (X-X.min())*(x_max-x_min)
    denom = X.max() - X.min()
    denom = denom + (denom is 0)
    return x_min + nom/denom


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
    '''if feedforward[0] > 0.70:
        print("It's a car!")
    else:
        print("It is not a car")'''
    return feedforward


def train(o, d, w1, w2, learnfactor):  # Takes (data(o[], label), d[], w1, w2, learnfactor)
    c = []
    z = []
    p = []
    n = []
    errsig1 = []
    weightgrad = []
    weightgrad2 = []
    congradient = []
    for g in range(len(o)):
        c.append(convolution(o[g][0], d)[0])
        z.append(maxpool(c[g]))                                    # z[0] is max pool, z[1] is location of maxes
        p.append(z[g][0].flatten())
        n.append(np.zeros((8649, 1)))

        for x in range(p[g].size):
            n[g][x] = (round(activation(p[g][x]), 2))           # list of input nodes after activation

        # print(n, "are the inputs to the net\n")            # Prints inputs into the dense network

        feedforward = np.matmul(w1, n[g])                   # Feed input through first layer
        for j in range(feedforward.size):
            feedforward[j] = activation(feedforward[j])

        actlayer1 = feedforward.copy()                       # Saves the output after activation for layer 1

        feedforward = np.matmul(w2, feedforward)             # Feed input through second layer
        for j in range(feedforward.size):
            feedforward[j] = activation(feedforward[j])

        saveoutput = feedforward.copy()

        error = np.zeros((2, 1))
        if o[g][1] == 1:                              # label == 1: training image is a car
            error[0] = feedforward[0] - 1
            error[1] = feedforward[1]
            e = (feedforward[0] - 1) ** 2 + (feedforward[1]) ** 2
        else:                                       # Else: Training image is a negative
            error[0] = feedforward[0]
            error[1] = feedforward[1] - 1
            e = (feedforward[0]) ** 2 + (feedforward[1] - 1) ** 2

        print("Error =", e, " ***************************")
        print(saveoutput)

        errsig2 = np.zeros((2, 1))                                   # Begin calculating new weights for w2
        for i in range(errsig2.size):
            errsig2[i][0] = error[i] * deriveact(saveoutput[i])

        weightgrad2.append(np.matmul(errsig2, np.transpose(actlayer1)))

        errsig1.append(np.zeros((1000, 1)))                                   # Begin calculating new weights for w1
        for i in range(errsig1[g].size):
            errsig1[g][i][0] = error[0] * deriveact(saveoutput[0]) + error[1] * deriveact(saveoutput[1])

        weightgrad.append(np.matmul(errsig1[g], np.transpose(n[g])))
# ///////////////////////////////////////////////////////////// stop loop here. next, calculate average
    layer2grad = np.zeros(weightgrad2[0].shape)
    layer1grad = np.zeros(weightgrad[0].shape)
    for j in range(len(weightgrad2)):                    # Finds the average of the weight gradients for all test images
        layer2grad = np.add(layer2grad, weightgrad2[j])
        layer1grad = np.add(layer1grad, weightgrad[j])
    layer2grad = np.divide(layer2grad, len(weightgrad2))
    layer1grad = np.divide(layer1grad, len(weightgrad))

    w2 = np.subtract(w2, layer2grad)                    # nudge the weights
    w1 = np.subtract(w1, layer1grad)

    # k_errsig = np.zeros(errsig1[0].shape)
    # for j in range(len(errsig1)):                       # finds the average gradient for the kernel
    #     k_errsig = np.add(k_errsig, errsig1[j])
    # k_errsig = np.divide(k_errsig, len(errsig1))

    for g in range(len(o)):                   # Walk through all image inputs after convolution, get avg kernel gradient
        wt = w1.transpose()
        wt = np.matmul(wt, errsig1[g])

        gradientmapin = np.zeros((8649, 1))

        for i in range(n[g].size):
            gradientmapin[i] = wt[i] * deriveact(n[g][i])

        revmp = np.zeros(c[g].shape)

        for i in range(gradientmapin.size):
            revmp[z[g][1][i][1]][z[g][1][i][0]] = gradientmapin[i]  # puts the gradients in place to reverse max pooling

        congradient.append(learnfactor * np.around(convolution(o[g][0], revmp)[0], 2))
# ////////////////////////////////////////////////////////////////// stop loop here
    avgcong = np.zeros(congradient[0].shape)
    for j in range(len(congradient)):                                  # calculate average grade for kernels
        avgcong = np.add(avgcong, congradient[j])
    avgcong = np.divide(avgcong, len(congradient))

    d = np.subtract(d, avgcong)                             # New Kernel

    return d, w1, w2


imageLocation = '\\Columbus_CSUAV_AFRL\\train'        # Where training images are stored relative to the CWD
o = []
n = []
(carpics, negpics) = getFileImages(os.getcwd() + imageLocation)
for i in range(100):
    img = Image.open(carpics[i]).convert('L')
    negimg = Image.open(negpics[i]).convert('L')
    o.append((scale(np.asarray(img), -1, 1), 1))
    n.append((scale(np.asarray(negimg), -1, 1), 0))

stddev = 1/np.sqrt(np.prod(4900))
d = []
for i in range(3):
    d.append(np.random.normal(loc=0, scale=stddev, size=4900))  # Initializes kernel randomly on a normal distribution
    d[i] = np.reshape(d[i], (70, 70))


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
'''
for i in range(1):
    for k in range(2):                                              # run through the training images
        for j in range(len(d)):                                     # Train all the filters'''
batchtest = []
batchtest.append(o[0])
batchtest.append(o[1])

d[0], w1, w2 = train(batchtest, d[0], w1, w2, learnfactor)        # Train a neg
d[0], w1, w2 = train(batchtest, d[0], w1, w2, learnfactor)        # Train a neg

#d[0], w1, w2 = train(n[0], d[0], w1, w2, learnfactor)        # Train a neg
'''d[0], w1, w2 = train(n[0], d[0], w1, w2, learnfactor, 0)        # Train a neg

d[0], w1, w2 = train(o[0], d[0], w1, w2, learnfactor, 1)        # Train a car
d[0], w1, w2 = train(o[0], d[0], w1, w2, learnfactor, 1)        # Train a car
d[0], w1, w2 = train(o[0], d[0], w1, w2, learnfactor, 1)        # Train a car'''





#print((j+1)*(k+1)*2, "/ 30 images trained")
'''
avg = np.zeros((2, 1))
for i in range(len(d)):
    avg = np.add(avg, run(n[70], d[i], w1, w2))

avg = np.divide(avg, len(d))
print("Error average:", avg)
if avg[0] > 0.70:
    print("It's a car!")
elif avg[0] < 0.30:
    print("It is not a car")
else:
    print("Undetermined")'''

# np.set_printoptions(threshold=sys.maxsize)

