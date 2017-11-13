# just gunna start over again. Have 8 hours

# lets see what needs to be done

#import images - DONE (and quite nicely if i do say so myself
# decimate function - DONE
# interpolate function
# Create lapl Pyramid
#   input is image
#   output is lapl and gaus pyramids



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












main()
