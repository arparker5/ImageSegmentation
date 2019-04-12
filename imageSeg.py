from PIL import Image
from skimage import data
from matplotlib import pyplot as plt
import numpy as np
import scipy as sp
import sys


def convolution(img, filter, s=1):
    (imgx, imgy) = img.shape
    (filtx, filty) = filter.shape

    outdim = int(imgx - filtx)+1
    outmtx = np.zeros(outdim, outdim)

    current_y = filty
    out_y = 0
    while imgy - current_y >= 0:
        current_x = filtx
        out_x = 0
        while imgx - current_x >= 0:
            cutout = img[current_y:current_y + filtx, current_x:current_x + filtx]
            outmtx[out_y, out_x] = filter * cutout
            current_x += 1
            out_x += 1
        current_y += 1
        out_y += 1


np.set_printoptions(threshold=sys.maxsize)
img = Image.open("imgtest.jpeg").convert('L')
img.save('greyscale.png')
x = np.asarray(img)

print(x.shape)

