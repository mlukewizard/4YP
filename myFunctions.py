from __future__ import division
from __future__ import print_function
#from keras.callbacks import ModelCheckpoint
#from keras.models import Model, load_model
#from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
#from keras.optimizers import Adam
#from keras import losses
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import scipy
from scipy import misc
import math
import os
import copy

def calcPerimeter(image):
    image = shell(image)
    whitePoints = np.array(np.where(np.isin(image, 255)))
    perimeter = 0
    continueSumming = True
    currentPoint = np.zeros([2, 1], dtype='int')
    currentPoint[0] = np.where(np.isin(image, 255))[0][0]
    currentPoint[1] = np.where(np.isin(image, 255))[1][0]
    while continueSumming:
        whitePoints = whitePoints - currentPoint
        distances = np.sqrt(np.square(whitePoints[0, :]) + np.square(whitePoints[1, :]))
        closestIndex = np.argmin(distances)
        perimeter = perimeter + min(distances)
        currentPoint[0] = whitePoints[0][closestIndex]
        currentPoint[1] = whitePoints[1][closestIndex]
        whitePoints = np.delete(whitePoints, closestIndex, 1)
        if len(whitePoints[0]) == 0:
            continueSumming = False
    return perimeter

def shell(image):
    image2 = getImagePerimeterPoints(image)
    xPoints = np.where(np.isin(image, 255))[0]
    yPoints = np.where(np.isin(image, 255))[1]
    for i, j in zip(xPoints, yPoints):
        conditionList = [image2[i, j-1] != 0, image2[i+1, j] != 0, image2[i, j+1] != 0, image2[i-1, j] != 0]
        if sum(conditionList) > 1:
            image2[i, j] = 0
    return image2

def calcSegmentLength(line):
    length = 0
    for j in range(line.shape[1]-1):
        length = length + np.sqrt(np.square(line[0, j+1] - line[0, j]) +
                        np.square(line[1, j+1] - line[1, j]) +
                        np.square(line[2, j+1] - line[2, j]))
    return length

def calc3DTortuosity(centreLine, windowLength):
    centreLineTortuosity = np.ndarray([centreLine.shape[1]])
    halfWindow = int(math.floor(windowLength/2))
    paddedCentreLine = np.pad(centreLine, halfWindow, 'edge')[halfWindow:-halfWindow, :]
    for i in range(centreLine.shape[1]):
        evalPoint = i + halfWindow
        absoluteDistance = np.sqrt(np.square(paddedCentreLine[0, evalPoint+halfWindow] - paddedCentreLine[0, evalPoint-halfWindow]) +
            np.square(paddedCentreLine[1, evalPoint + halfWindow] - paddedCentreLine[1, evalPoint - halfWindow]) +
            np.square(paddedCentreLine[2, evalPoint + halfWindow] - paddedCentreLine[2, evalPoint - halfWindow]))
        curveDistance = 0
        for j in range(windowLength-1):
            curveDistance = curveDistance + np.sqrt(np.square(paddedCentreLine[0, i+j+1] - paddedCentreLine[0, i+j]) +
                                                    np.square(paddedCentreLine[1, i+j+1] - paddedCentreLine[1, i+j]) +
                                                    np.square(paddedCentreLine[2, i+j+1] - paddedCentreLine[2, i+j]))
        tortuosity = curveDistance / absoluteDistance
        centreLineTortuosity[i] = tortuosity
    return centreLineTortuosity

def findAAABounds(wallVolume, OuterDiameter):
    maxDia = len(OuterDiameter[0:-100]) + np.argmax(OuterDiameter[-100:])
    maxVol = len(wallVolume[0:-100]) + np.argmax(wallVolume[-100:])
    acceptableSteps = np.linspace(1, 50, 50, dtype='int').tolist()
    timeSerieses = [wallVolume, OuterDiameter, wallVolume, OuterDiameter]
    directions = [-1, -1, 1, 1]
    points = []
    maxLocations = [maxVol, maxDia, maxVol, maxDia]
    for timeSeries, direction, start in zip(timeSerieses, directions, maxLocations):
        looking = True
        copyStart = copy.deepcopy(start)
        while looking:
            if any(((np.append(timeSeries, np.zeros(60, dtype = 'int'))[copyStart + direction*step] - np.append(timeSeries, np.zeros(60, dtype = 'int'))[copyStart])/step < -np.max(timeSeries)/200 and np.append(timeSeries, np.zeros(60, dtype = 'int'))[copyStart + direction*step] != 0) for step in acceptableSteps):
                copyStart = copyStart + direction
            else:
                looking = False
        points.append(copyStart)
    if abs(points[3] - points[2]) > 10:
        print('Warning: Your algorithm is unsure about where the aneurysm ends, wall thickness suggests ' + str(points[2]) + ' and diameter suggests ' + str(points[3]))
    if abs(points[1] - points[0]) > 10:
        print('Warning: Your algorithm is unsure about where the aneurysm starts, wall thickness suggests ' + str(points[0]) + ' and diameter suggests ' + str(points[1]))
    return [int((points[0] + points[1])/2), int((points[2] + points[3])/2)]

def isDoubleAAA(image):
    OuterTop = np.array(np.where(np.isin(image, 255)))[:, 0]
    OuterBottom = np.array(np.where(np.isin(image, 255)))[:, -1]
    continueLoop = True
    while continueLoop:
        x = (OuterTop - OuterBottom)/np.max(abs((OuterBottom - OuterTop)))
        if image[OuterBottom[0] + int(x[0]), OuterBottom[1] + int(x[1])] == 255 or image[OuterBottom[0] + 2*int(x[0]), OuterBottom[1] + 2*int(x[1])] == 255 or image[OuterBottom[0] + 5*int(x[0]), OuterBottom[1] + 5*int(x[1])] == 255:
            OuterBottom[0] = OuterBottom[0] + int(x[0])
            OuterBottom[1] = OuterBottom[1] + int(x[1])
        else:
            return True
        if np.square(OuterTop[0] - OuterBottom[0]) + np.square(OuterTop[1] - OuterBottom[1]) < 400:
            return False

def ConstructArraySlice(inputFolder1, inputFolder1Dir, inputFileIndex, boxSize,inputFolder2=None, inputFolder2Dir= 'Blank', centralLocation=None, twoDVersion = False):
    import scipy.misc as misc
    import sys
    import os
    import dicom
    import PIL
    from PIL import Image
    import matplotlib.pyplot as plt
    from uuid import getnode as get_mac
    mac = get_mac()
    if mac != 176507742233701:
        tmpFolder = 
    else:
        tmpFolder = 'C:\\Users\\Luke\\Documents\\sharedFolder\\4YP\\4YP_Python\\tmp\\'
    if not os.path.exists(tmpFolder):
        os.mkdir(tmpFolder)
    dicomSlice = False

    # Just a little check
    if inputFolder2!= None and len(inputFolder2) != len(inputFolder1):
        sys.exit('Your folders arent the same length, something is seriously wrong')

    # Making the distinction between binary and dicom a little easier
    if inputFolder2 == None:
        dicomSlice = True

    if dicomSlice == True:
        fileLists = [inputFolder1]
        image1Files = []
        doc = [image1Files]
    else:
        image1Files = []
        image2Files = []
        fileLists = [inputFolder1, inputFolder2]
        doc = [image1Files, image2Files]

    if not twoDVersion:
        arrayIndexes = np.linspace(inputFileIndex-6, inputFileIndex+6, 5, dtype='int')
        for i in range(len(arrayIndexes)):
            if arrayIndexes[i] < 0:
                arrayIndexes[i] = 0
            elif arrayIndexes[i] > len(inputFolder1)-1:
                arrayIndexes[i] = len(inputFolder1)-1
    else:
        arrayIndexes = [inputFileIndex]
    #print(arrayIndexes)
    for imageFiles, inputFolder, fileList, inputFolderDir in zip(doc, [inputFolder1, inputFolder2], fileLists, [inputFolder1Dir, inputFolder2Dir]):
        for i in range(len(arrayIndexes)):
            if '.' in fileList[arrayIndexes[i]]:
                imageFiles.append(np.array(Image.open(inputFolderDir + fileList[arrayIndexes[i]]).convert('F')))
            else:
                dicomImage = dicom.read_file(inputFolderDir + fileList[arrayIndexes[i]]).pixel_array
                misc.imsave(tmpFolder + 'dicomTemp.png', dicomImage)
                imageFiles.append(misc.imread(tmpFolder + 'dicomTemp.png'))
                os.remove(tmpFolder + 'dicomTemp.png')
        if (not imageFiles[0].shape[0] == boxSize or not imageFiles[0].shape[1] == boxSize) and (centralLocation is None):
            sys.exit('If your image isnt 144x144 then you need to tell me the central location')
        elif (not imageFiles[0].shape[0] == boxSize or not imageFiles[0].shape[1] == boxSize) and (centralLocation is not None):
            upperRow = centralLocation[0] - round(boxSize/2)
            lowerRow = upperRow + boxSize
            leftColumn = centralLocation[1] - round(boxSize/2)
            rightColumn = leftColumn + boxSize
            for i in range(len(arrayIndexes)):
                imageFiles[i] = imageFiles[i][upperRow:lowerRow, leftColumn:rightColumn]

    if not twoDVersion:
        if dicomSlice:
            slice = np.ndarray((len(arrayIndexes), boxSize, boxSize, 1), dtype='float32')
            for i in range(len(arrayIndexes)):
                slice[i, :, :, 0] = image1Files[i]
            return slice
        else:
            slice = np.ndarray((len(arrayIndexes), boxSize, boxSize, 2), dtype='float32')
            for i in range(len(arrayIndexes)):
                slice[i, :, :, 0] = image1Files[i]
                slice[i, :, :, 1] = image2Files[i]
            #saveSlice(slice, showFig=True)
            return slice
    else:
        if dicomSlice:
            slice = np.ndarray((boxSize, boxSize, 1), dtype='float32')
            slice[:, :, 0] = image1Files[i]
            return slice
        else:
            slice = np.ndarray((boxSize, boxSize, 2), dtype='float32')
            slice[:, :, 0] = image1Files[i]
            slice[:, :, 1] = image2Files[i]
            return slice

def findLargestNumberInString(text):
    li = [0]
    for i in range(len(text)):
        num = ""
        if text[i].isdigit():
            while text[i].isdigit():
                num = num + text[i]
                i = i + 1
            li.append(int(num))
    return max(li)

def findLargestNumberInFolder(list):
    import types
    largestNum = 0
    for i in range(len(list)):
        if (findLargestNumberInString(list[i]) > largestNum):
            largestNum = findLargestNumberInString(list[i])
    return largestNum

def findSmallestNumberInString(text):
    li = []
    for i in range(len(text)):
        num = ""
        if text[i].isdigit():
            while text[i].isdigit():
                num = num + text[i]
                i = i + 1
            li.append(int(num))
    return min(li)

def findSmallestNumberInFolder(list):
    smallestNum = 10000
    for i in range(len(list)):
        if (findSmallestNumberInString(list[i]) < smallestNum):
            smallestNum = findSmallestNumberInString(list[i])
    return smallestNum

def getImagePerimeterPoints(inputImage):
    image = Image.fromarray(inputImage)
    image = image.filter(ImageFilter.FIND_EDGES)
    outputImage = np.array(image)
    return outputImage

def getImageBoundingBox(inputImage):
    from scipy import ndimage
    import numpy as np

    if np.max(inputImage) > 0:
        return np.array([min(np.where(np.isin(np.transpose(inputImage), 255))[0]), max(np.where(np.isin(inputImage, 255))[1]), min(np.where(np.isin(inputImage, 255))[0]), max(np.where(np.isin(np.transpose(inputImage), 255))[1])])
    else:
        return np.array([256, 256, 256, 256])

def getFolderBoundingBox(filePath):
    import os
    import numpy as np
    from scipy import misc
    cumulativeImage = np.zeros(shape=(512,512), dtype='float32')
    fileList = sorted(os.listdir(filePath))

    for filename in fileList:
        newFile = np.array(misc.imread(filePath + filename, flatten=True))
        cumulativeImage = np.add(cumulativeImage, newFile)

    return getImageBoundingBox(cumulativeImage)

def getFolderCoM(dicomFolder):
    import dicom
    import os
    from scipy import ndimage
    import numpy as np
    import math

    inputImage = np.ndarray([512, 512])
    fileList = sorted(os.listdir(dicomFolder))
    sampleFileList = filter(lambda k: '80' in k, fileList)
    i = 0
    xTotal = 0
    yTotal = 0
    for filename in sampleFileList:
        i = i + 1
        image = dicom.read_file(dicomFolder + filename)
        inputImage[:, :] = image.pixel_array

        # Gets image centre of mass, note y coordinate comes first and then x coordinate
        CoM = ndimage.measurements.center_of_mass(inputImage)
        xTotal = xTotal + CoM[1]
        yTotal = yTotal + CoM[0]

    xAvg = math.floor(xTotal / i)
    yAvg = math.floor(yTotal / i)

    # Sets the limits for a 256x256 bounding box
    xMin = int(xAvg - 128 if xAvg - 128 > 0 else 0)
    xMax = int(xMin + 256)
    yMin = int(yAvg - 128 if yAvg - 128 > 0 else 0)
    yMax = int(yMin + 256)
    return np.array([xMin, xMax, yMin, yMax])

def lukesAugment(image):
    import numpy as np

    def f(x):
            return 135.6*np.tanh((x-150)/70) + 132

    f = np.vectorize(f)  # or use a different name if you want to keep the original f
    image = f(image)
    return image

def saveSlice(slice1, slice2=np.zeros(7), saveFig = False, showFig = False, saveFolder = ''):
    import matplotlib.pyplot as plt
    import numpy as np
    import datetime
    import sys
    counter = 0
    mySlices = [slice1]
    if not slice2.shape[0] == 7:
        mySlices.append(slice2)
    totalDepth = 0
    for slice in mySlices:
        if slice.shape[0] > 10:
            sliceDepth = slice.shape[2]
            totalDepth = totalDepth + sliceDepth
        else:
            sliceDepth = slice.shape[3]
            totalDepth = totalDepth + sliceDepth
    for slice in mySlices:
        if slice.shape[0] > 10:
            sliceWidth = 1
            sliceDepth = slice.shape[2]
            twoD = True
        else:
            sliceWidth = slice.shape[0]
            sliceDepth = slice.shape[3]
            twoD = False
        for j in range(sliceDepth):
            for i in range(sliceWidth):
                plt.subplot(totalDepth, sliceWidth, counter+1)
                if not twoD:
                    plt.imshow(slice[i, :, :, j], cmap='gray')
                else:
                    plt.imshow(slice[:, :, j], cmap='gray')
                plt.axis('off')
                counter = counter + 1

    if showFig == True:
        mng = plt.get_current_fig_manager()
        mng.window.state('zoomed')
        plt.show()
    if saveFig == True:
        if saveFolder == '':
            sys.exit('If you set saveFig as true you also need to specify saveFolder')
        else:
            plt.savefig(saveFolder + 'Figure_' + str(datetime.datetime.now()).split('.')[0].replace(':','-').replace(' ','_'))
