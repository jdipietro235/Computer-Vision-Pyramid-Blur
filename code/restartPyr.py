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
            #imageAName = raw_input('Image A filename\n> ')
            imageAName = 'apple.jpg'
            imageA = misc.imread(imageAName, flatten=1)
            imageA = imageA.astype(float)
            break
        except(IOError):
            print('invalid name')

    while(True):
        try:
            #imageBName = raw_input('Image B filename\n> ')
            imageBName = 'orange.jpg'
            imageB = misc.imread(imageBName, flatten=1)
            imageB = imageB.astype(float)
            break
        except(IOError):
            print('invalid name')

    kernel = (1.0/256)*np.array([[1, 4,  6,  4,  1],
                                 [4, 16, 24, 16, 4],
                                 [6, 24, 36, 24, 6],
                                 [4, 16, 24, 16, 4],
                                 [1, 4,  6,  4,  1]])

    

    #gausA, laplPyrA = pyramids(imageA, kernel, 2)
    #gausB, laplPyrB = pyramids(imageB, kernel, 2)

    pyrA = pyramids(imageA, kernel, 2)
    pyrB = pyramids(imageB, kernel, 2)

    joinedPyr = halves(pyrA, pyrB)

    collapsed = collapse(joinedPyr, kernel)

    rowCountA, colCountA = imageA.shape
    rowCountB, colCountB = imageB.shape
    
def collapse(joinedPyr, kernel):
    print('collapse')
    
    output = np.zeros((joinedPyr[0].shape[0],joinedPyr[0].shape[1]), dtype=np.float64)
    for i in range(len(joinedPyr)-1,0,-1):
        
        #we're expanding it
        focusLevel = joinedPyr[i]
        focus2 = np.zeros((focusLevel.shape[0]*2, focusLevel.shape[1]*2), dtype=np.float64)
        focus2[::2,::2] = focusLevel[:,:]
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


def halves(glPyrA, glPyrB):
    print('halves start')
    output = []
    print(type(glPyrA))
    #print(laplPyrA.shape)
    zipped = (glPyrA, glPyrB)
    i = 0
    while i < len(zipped):
        print('halves looping')
        laplA = np.asarray(glPyrA[1][i])
        laplB = np.asarray(glPyrB[1][i])
        #print(laplA)
        #laplA = np.as2darray(laplA)
        rowCount, colCount = laplA.shape
        layer = np.stack((laplA[:,0:colCount/2], laplB[:, colCount/2:]))
        output.append(layer)
        i += 1
    return output
    '''
    for laplA, laplB in zipped:
        print('halves looping')
        rowCount, colCount, depthCount = laplA.shape
        layer = np.stack((laplA[:,0:colCount/2], laplB[:, colCount/2:]))
        output.append(layer)
        '''


def flux(gaus, laplPyr):

    #gaus, laplPyr = pyramids(image, kernel, rate)

    joinedImage = np.zeros((rowCount, colCount + colCount / 2), dtype=np.double)
    joinedImage[:rowCount, :colCount] = gaus[0]

    iRow = 0
    for p in gaus[1:]:
        nRows, nCols = p.shape[:2]
        joinedImage[iRow:iRow +nRows, colCount:colCount + nCols] = p
        iRow += nRows

    
    fig, ax = plt.subplots()

    ax.imshow(joinedImage, cmap = 'gray')
    plt.show()
    

    #return gaus,laplPyr



def decimate(image, kernel, rate):
    imageBlurred = ndimage.filters.convolve(image, kernel, mode = 'constant')
    imageDecimated = imageBlurred[::rate, ::rate] #keeps every (rate) pixel

    return imageDecimated


def interpolate(image, kernel, rate):
    newImage = np.zeros((rate*image.shape[0], rate*image.shape[1]))
    newImage[::rate, ::rate] = image

    dRate = rate*2
    output = ndimage.filters.convolve(newImage, dRate*kernel, mode='constant')

    return output


def pyramids(image, kernel, rate):
    gausPyr = [image, ]
    laplPyr = []

    modImage = image

    while modImage.shape[0] >= 2 and modImage.shape[1] >= 2:
        modImage = decimate(modImage, kernel, rate)
        gausPyr.append(modImage)

    for i in range(len(gausPyr) - 1):
        gausInter = interpolate(gausPyr[i+1], kernel, rate)
        laplPyr.append(gausPyr[i] - gausInter)
        #the lapl level is a level of gaus minus the next level up

    return gausPyr[:-1], laplPyr #:-1









main()
