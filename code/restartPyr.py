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

    

    print imageA.shape
    #gausA, laplPyrA = pyramids(imageA, kernel, 2)
    #gausB, laplPyrB = pyramids(imageB, kernel, 2)

    pyrA = pyramids(imageA, kernel, 2)
    pyrB = pyramids(imageB, kernel, 2)

    print pyrA[1][1].shape

    joinedPyr = halves(pyrA, pyrB)

    collapsed = smush(joinedPyr, kernel, 2)

    fig, ax = plt.subplots()
        
    ax.imshow(collapsed,cmap='gray')
    plt.show()    


def smush(joinedPyr, kernel, rate):
    #take the smallest level of the pyramid
    # expand it to match the next larger
    #
    

    #create an empty image to hold smushed stuff
    #this is pointing at joinedPyr[0] bc thats the bottom layer

    print('joinedPyr[0] shape')
    print(joinedPyr[0].shape)
    
    output = np.zeros((joinedPyr[0].shape[0], joinedPyr[0].shape[1]), dtype=np.float64)

    print('pyramid sizes')
    print joinedPyr[0].shape
    print joinedPyr[1].shape
    #print joinedPyr[2].shape

    i = len(joinedPyr)
    
        
    #for i in reversed(xrange(len(joinedPyr))):
    while i > 1:
        print len(joinedPyr)
        layer = joinedPyr[i-1]

        smushedLayer = interpolate(layer, kernel, rate)

        print('smushedlayer size')
        print smushedLayer.shape


        '''
        smushedLayer = np.zeros((layer.shape[1]*2, layer.shape[2]*2))
        smushedLayer[::rate, ::rate] = layer[:,:]
    # sets an image to zeros. Sets every other pixel to the pixels of og image

        smushedLayer = ndimage.filters.convolve(smushedLayer, 4*kernel, mode='constant')
        '''
        #smushedLayer = np.zeros((layer.shape[0]*2, layer.shape[1]*2, layer.shape[2]), dtype=np.float64) #so now we double the size of the thing.
        #depth might actually be first value, which might be the problem

        # the process that im going thru rn looks a lot like interpolate

        backLayer = joinedPyr[i-2] #-1
        '''
        if smushedLayer.shape[0] > backLayer.shape[0]:
            smushedLayer = np.delete(smushedLayer,(-1),axis=0)
            
        if smushedLayer.shape[1] > backLayer.shape[1]:
            smushedLayer = np.delete(smushedLayer,(-1),axis=1)
        '''        

        tmp = smushedLayer + backLayer
        
        joinedPyr.pop()
        joinedPyr.pop()
        joinedPyr.append(tmp)
        output = tmp
        i -= 1
    return output
        
    
def collapse(joinedPyr, kernel):
    print('collapse')

    np.set_printoptions(threshold='nan')
    
    output = np.zeros((joinedPyr[0].shape[0],joinedPyr[0].shape[1]), dtype=np.float64)
    for i in range(len(joinedPyr)-1,0,-1):
        
        #we're expanding it
        focusLevel = joinedPyr[i]
        mergedLevel = np.zeros((focusLevel.shape[0]*2, focusLevel.shape[1]*2, focusLevel.shape[2]), dtype=np.float64)

        print i
        #print focusLevel
        print('mergedLevel')

        #print mergedLevel
        mergedLevel[::2,::2] = focusLevel[:,:]

        focusA = 4*scipy.signal.convolve2d(mergedLevel,kernel,'same')#convolve2d
        
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
        laplA = np.asarray(glPyrA[1][i]) #glPyrA[1][i]
        print'laplA.shape'
        print laplA.shape
        laplB = np.asarray(glPyrB[1][i])
        #print(laplA)
        #laplA = np.as2darray(laplA)
        rowCount, colCount = laplA.shape
        layer = np.hstack((laplA[:,0:colCount/2], laplB[:, colCount/2:]))
        print layer.shape
        output.append(layer)
        i += 1
    print output[0].shape
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
