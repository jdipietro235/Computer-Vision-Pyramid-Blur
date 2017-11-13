# just gunna start over again. Have 8 hours

# lets see what needs to be done

#import images - DONE (and quite nicely if i do say so myself


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



















main()
