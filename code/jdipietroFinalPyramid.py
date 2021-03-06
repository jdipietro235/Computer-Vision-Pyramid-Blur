
import numpy as np
import scipy
import scipy.signal as sig
from scipy import misc
import matplotlib.pyplot as plt
from scipy import ndimage
import cv2

# Justin DiPietro
# Computer Vision Assignement 4



def main():
    while(True):
        try:
            #imageAName = raw_input('Image A filename\n> ')
            imageAName = 'house.png'
            imageA = misc.imread(imageAName, flatten=0)
            imageA = imageA.astype(float)
            break
        except(IOError):
            print('invalid name')

    while(True):
        try:
            #imageBName = raw_input('Image B filename\n> ')
            imageBName = 'castle.png'
            imageB = misc.imread(imageBName, flatten=0)
            imageB = imageB.astype(float)
            break
        except(IOError):
            print('invalid name')

    kernel = (1.0/256)*np.array([[1, 4,  6,  4,  1],
                                 [4, 16, 24, 16, 4],
                                 [6, 24, 36, 24, 6],
                                 [4, 16, 24, 16, 4],
                                 [1, 4,  6,  4,  1]])


    aBlue, aGreen, aRed = imageA[:, :, 0], imageA[:, :, 1], imageA[:, :, 2]
    bBlue, bGreen, bRed = imageB[:, :, 0], imageB[:, :, 1], imageB[:, :, 2]


    pyrAblue = pyramids(aBlue, kernel, 2)
    pyrAgreen = pyramids(aGreen, kernel, 2)
    pyrAred = pyramids(aRed, kernel, 2)

    pyrBblue = pyramids(bBlue, kernel, 2)
    pyrBgreen = pyramids(bGreen, kernel, 2)
    pyrBred = pyramids(bRed, kernel, 2)

    joinedPyrBlue = halves(pyrAblue, pyrBblue)
    joinedPyrGreen = halves(pyrAgreen, pyrBgreen)
    joinedPyrRed = halves(pyrAred, pyrBred)

    collapsedBlue = collapse(joinedPyrBlue, kernel, 2)
    collapsedGreen = collapse(joinedPyrGreen, kernel, 2)
    collapsedRed = collapse(joinedPyrRed, kernel, 2)

    collapsedBlue = collapsedBlue[..., np.newaxis]
    collapsedGreen = collapsedGreen[..., np.newaxis]
    collapsedRed = collapsedRed[..., np.newaxis]

    stacked = collapsedBlue, collapsedGreen, collapsedRed
    end_img = np.dstack(stacked)

    #cv2.imwrite('house_castle_badCode.jpg',end_img)

    
    fig, ax = plt.subplots()
        
    ax.imshow(end_img)
    plt.show()
    


def collapse(joinedPyr, kernel, rate):
    #take the smallest level of the pyramid
    # expand it to match the next larger
    #create an empty image to hold smushed stuff
    #this is pointing at joinedPyr[0] bc thats the bottom layer
    
    output = np.zeros((joinedPyr[0].shape[0], joinedPyr[0].shape[1]), dtype=np.float64)
    
    i = len(joinedPyr)
    while i > 1:
        layer = joinedPyr[i-1]

        smushedLayer = interpolate(layer, kernel, rate)

        backLayer = joinedPyr[i-2] #-1
            
        tmp = smushedLayer + backLayer
        
        joinedPyr.pop()
        joinedPyr.pop()
        joinedPyr.append(tmp)
        output = tmp
        i -= 1
    return output


def halves(glPyrA, glPyrB):
    output = []
    zipped = (glPyrA, glPyrB)
    i = 0
    while i < len(zipped):
        laplA = np.asarray(glPyrA[1][i]) 
        laplB = np.asarray(glPyrB[1][i])

        rowCount, colCount = laplA.shape
        layer = np.hstack((laplA[:,0:colCount/2], laplB[:, colCount/2:]))
        
        output.append(layer)
        i += 1
    print output[0].shape
    return output


def flux(gaus, laplPyr):

    #gaus, laplPyr = pyramids(image, kernel, rate)
    rowCount, colCount = laplPyr[0].shape

    compoundImg = np.zeros((rowCount, colCount + colCount / 2), dtype=np.double)
    compoundImg[:rowCount, :colCount] = gaus[0]

    iRow = 0
    for p in gaus[1:]:
        nRows, nCols = p.shape[:2]
        joinedImage[iRow:iRow +nRows, colCount:colCount + nCols] = p
        iRow += nRows

    
    fig, ax = plt.subplots()

    ax.imshow(compoundImg, cmap = 'gray')
    plt.show()

    cv2.imwrite('house_castle_compound.jpg',compoundImg)

    return compoundImg



def decimate(image, kernel, rate):
    imageBlurred = ndimage.filters.convolve(image, kernel, mode = 'constant')
    imageDecimated = imageBlurred[::rate, ::rate] #keeps every other(rate) pixel

    return imageDecimated


def interpolate(image, kernel, rate): # returns and image of 2x the size of input
    newImage = np.zeros((image.shape[0]*rate, image.shape[1]*rate))
    newImage[::rate, ::rate] = image[:,:]
    # sets an image to zeros. Sets every other pixel to the pixels of og image

    dRate = rate*2
    output = ndimage.filters.convolve(newImage, dRate*kernel, mode='constant')
    # blurs the out image so that it doesnt have white pixels all over the place

    return output


def pyramids(image, kernel, rate):
    gausPyr = [image, ]
    laplPyr = []

    while image.shape[0] >= 2 and image.shape[1] >= 2:
        image = decimate(image, kernel, rate)
        gausPyr.append(image)

    for i in range(len(gausPyr) - 1):
        gausInter = interpolate(gausPyr[i+1], kernel, rate)
        laplPyr.append(gausPyr[i] - gausInter)
        #the lapl level is a level of gaus minus the next level up

    return gausPyr[:-1], laplPyr #:-1



main()
