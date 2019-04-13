from PIL import Image
from skimage import data
from matplotlib import pyplot as plt
import numpy as np
import scipy as sp
import sys


def convolution(img, filter, s=1):
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
            outmtx[out_y, out_x] = a.sum()
            current_x += 1
            out_x += 1
        current_y += 1
        out_y += 1

    return outmtx


def maxpool(img, filt=2, s=2):
    (imgy, imgx) = img.shape

    outdim_x = int((imgx - filt)/s) + 1
    outdim_y = int((imgy - filt)/s) + 1

    outmtx = np.zeros((outdim_y, outdim_x))

    current_y = 0
    out_y = 0
    while current_y + filt <= imgy:
        current_x = 0
        out_x = 0
        while current_x + filt <= imgx:
            a = img[current_y:current_y + filt, current_x:current_x + filt]
            outmtx[out_y, out_x] = np.max(a)
            current_x += s
            out_x += 1
        current_y += s
        out_y += 1

    return outmtx



np.set_printoptions(threshold=sys.maxsize)
img = Image.open("pixilimg.png").convert('L')
img.save('imggrey.png')
imgg = np.asarray(img)

filt = Image.open("filter.png").convert('L')
filt.save('filtergrey.png')
filtg = np.asarray(filt)


conimg = convolution(imgg, filtg)

mp = maxpool(conimg)

print(conimg.shape)
print(mp.shape)

