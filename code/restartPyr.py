# just gunna start over again. Have 8 hours

# lets see what needs to be done

#import images - DONE (and quite nicely if i do say so myself
# decimate function - DONE
# interpolate function
# Create lapl Pyramid
#   input is image
#   output is lapl and gaus pyramids

import numpy as np
import scipy
import scipy.signal as sig
from scipy import misc
import matplotlib.pyplot as plt
from scipy import ndimage

def main():
    while(True):
        try:
            imageAName = raw_input('Image A filename\n> ')
            imageA = misc.imread(imageAName)
            break
        except(IOError):
            print('invalid name')

    while(True):
        try:
            imageBName = raw_input('Image B filename\n> ')
            imageB = misc.imread(imageBName)
            break
        except(IOError):
            print('invalid name')

    gausA, laplPyrA = pyramids(imageA)
    gausB, laplPyrB = pyramids(imageB)

    rowCountA, colCountA = imageA.shape
    rowCountB, colCountB = imageB.shape

    compositeImage = np.zeros((rowCountA, colCountA + colCountA / 2), dtype=np.double)
    compositeImage[:rowCountA, :colCountA] = gausA[0]



def decimate(image, kernel, rate):
    imageBlurred = ndimage.filters.convolve(image, kernel, mode = 'constant')
    imageDecimated = imageBlurred[::rate, ::rate] #keeps every (rate) pixel

    return imageDecimated


def interpolate(image, kernel, rate):
    newImage = np.zeros(rate*image.shape[0], rate*image.shape[1])
    newImage[::rate, ::rate] = image

    dRate = rate*2
    output = ndimage.filters.convolve(newImage, dRate*kernel, mode='constant')

    return output


def pyramids(image, kernel, rate):
    gausPyr = [image, ]
    laplPyr = []

    modImage = image

    while modImage.shape[0] >= 2 and modImage.shape[1] >= 2:
        modImage = decimate(modImage)
        gausPyr.append(modImage)

    for i in range(len(GausPyr) - 1):
        laplPyr.append(gausPyr[i] - interpolate(gaus[i + 1]))
        #the lapl level is a level of gaus minus the next level up

    return gausPyr[:-1], laplPyr









main()
