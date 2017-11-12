# This is my pyramid image blender

"""
The objective of this program is to create blend together 2 split images along
a central axis
I will do this with pyramids. Not sure how to yet


"""

import numpy as np
import scipy.signal as sig
from scipy import misc
import matplotlib.pyplot as plt
from scipy import ndimage
import scipy
import cv2


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

    # This is the Binomial (5-tap) filter
    kernel = (1.0/256)*np.array([[1, 4,  6,  4,  1],
                                 [4, 16, 24, 16, 4],
                                 [6, 24, 36, 24, 6],
                                 [4, 16, 24, 16, 4],
                                 [1, 4,  6,  4,  1]])

    # this displays the kernel
    #plt.imshow(kernel)
    #plt.show()

    '''
    aBlue, aGreen, aRed = imageA[:, :, 0], imageA[:, :, 1], imageA[:, :, 2]
    bBlue, bGreen, bRed = imageB[:, :, 0], imageB[:, :, 1], imageB[:, :, 2]
    '''


    pyrA = eacher(imageA, kernel)


    pyrB = eacher(imageB, kernel)


    joinedPyr = halves(pyrA, pyrB)


    collapsed = collapse(joinedPyr, kernel)



    fig, ax = plt.subplots()

    ax.imshow(end_img, cmap = 'gray')
    plt.show()


def collapse(joinedPyr, kernel):
    print('collapse')
    
    output = np.zeros((joinedPyr[0].shape[0],joinedPyr[0].shape[1]), dtype=np.float64)
    for i in range(len(joinedPyr)-1,0,-1):
        
        #we're expanding it
        focusLevel = joinedPyr[i]
        focus2 = np.zeros((focusLevel.shape[0]*2, focusLevel.shape[1]*2), dtype=np.float64)
        focus2[::2,::2]=focusLevel[:,:]
        focusA = 4*scipy.signal.convolve2d(focus2,kernel,'same')
        
        focusB = joinedPyr[i-1]
        if focusA.shape[0] > focusB.shape[0]:
            focusA = np.delete(focusA,(-1),axis=0)
        if focusA.shape[1] > focusB.shape[1]:
            lap = np.delete(lap,(-1),axis=1)
        tmp = focusA + focusB
        joinedPyr.pop()
        joinedPyr.pop()
        joinedPyr.append(tmp)
        output = tmp
    return output





def halves(lPyrA, lPyrB):
    print('halves start')
    LS = []
    zipped = (lPyrA, lPyrB)
    print('zipped length: ' + str(len(zipped)))
    i = 0
    while i < len(zipped):
    #for laplA, laplB in zip(lPyrA, lPyrB):
        laplA = lPyrA[i]
        laplB = lPyrB[i]
        print('Halves looping')
        #print(laplA)
        laplA = np.asarray(laplA)
        print('laplA shape: ' + str(laplA.shape))
        #laplA = laplA[..., np.newaxis]
        rowCount, colCount = laplA.shape #depthCount colCount
        ls = np.stack((laplA[:,0:colCount/2], laplB[:, colCount/2:]))
        LS.append(ls)
        i += 1
    return LS

def eacher(image, kernel):
    print('eacher start')
    [gLevel, lPyr] = pyramids(image, kernel)

    rowCount, colCount = image.shape

    joinedImage = np.zeros((rowCount, colCount + colCount / 2), dtype = np.double)
    joinedImage[:rowCount, :colCount] = gPyr[0]

    iRow = 0
    for p in gLevel:  #[1:]
        nRows, nCols = p.shape[:2]
        joinedImage[iRow:iRow +nRows, colCount:colCount + nCols] = p
        iRow += nRows
    '''
    fig, ax = plt.subplots()

    ax.imshow(joinedImage, cmap = 'gray')
    plt.show()
    '''
    return gLevel,lPyr


def pyramids(image, kernel):  # break this into 2 functions
    gPyr = [image, ]
    lPyr = []

    while image.shape[0] >= 2 and image.shape[1] >= 2:
        image = decimate(image, kernel, 2)
        gPyr.append(image)

    for i in range(len(gPyr) - 1):
        lPyr.append(gPyr[i] - interpolate(gPyr[i + 1],kernel, 2))

    return gPyr[:-1], lPyr

'''
def createGausPyr(image, kernel):
    gPyr = [image, ]
    print('createGausPyr')
    while image.shape[0] >= 2 and image.shape[1] >= 2:
        image = decimate(image, kernel, 2)
        gPyr.append(image)

def createLaplPyr(gPyr):
    lPyr = []
    for i in range(len(gPyr) - 1):
        lPyr.append(gPyr[i] - interpolate(gPyr[i + 1],kernel, 2))
'''

def decimate(image, kernel, rate): # This is where we take only some of the pixels
    # It makes the image smaller
    # It is blurred to prevent bad stuff
    #rate is norammly 2
    imageBlurred = ndimage.filters.convolve(image, kernel, mode = 'constant')
    dSampled = imageBlurred[::rate, ::rate]
    return dSampled


def interpolate(image, kernel, rate):
    imageUp = np.zeros((rate*image.shape[0], rate*image.shape[1]))
    imageUp[::rate, ::rate] = image

    return ndimage.filters.convolve(imageUp, 4*kernel, mode='constant')
    


main()
