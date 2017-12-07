from __future__ import division
from __future__ import print_function
from keras.callbacks import ModelCheckpoint
from keras.models import Model, load_model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from keras.optimizers import Adam
from keras import losses
import numpy as np
import h5py
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance, ImageFilter

def getImagePerimeterPoints(inputImage):

    '''
    outputImage = np.zeros(shape = (inputImage.shape[0], inputImage.shape[1]))
    for i in range(inputImage.shape[0]):
        foundLeftEdge = False
        foundRightEdge = False
        for j in range(inputImage.shape[1]):
            if (foundLeftEdge == False) & inputImage[i,j] != 0:
                outputImage[i,j] = 1
                foundLeftEdge = True
                break
            if (foundLeftEdge == True) and (foundRightEdge == False) and inputImage[i,j] == 0:
                outputImage[i, j] = 1
                foundRightEdge = True
                break

    for j in range(inputImage.shape[0]):
        foundTopEdge = False
        foundBottomEdge = False
        for i in range(inputImage.shape[1]):
            if (foundTopEdge == False) & inputImage[i,j] != 0:
                outputImage[i,j] = 1
                foundTopEdge = True
                break
            if (foundTopEdge == True) and (foundBottomEdge == False) and inputImage[i,j] == 0:
                outputImage[i, j] = 1
                foundBottomEdge = True
                break
    '''
    image = Image.fromarray(inputImage)
    image = image.filter(ImageFilter.FIND_EDGES)
    outputImage = np.array(image)

    return outputImage

def getImageBoundingBox(inputImage):
    from scipy import ndimage
    import numpy as np

    image = ndimage.gaussian_filter(inputImage, sigma=1)

    minX = 500
    maxX = 10
    maxY = 10
    minY = 500
    for i in range(image.shape[1]):
        for j in range(image.shape[0]):
            if (image[j,i] > 200):
                if (i < minX):
                    minX = i
                elif (i > maxX):
                    maxX = i
                if (j < minY):
                    minY = j
                elif (j > maxY):
                    maxY = j
    return np.array([minX, maxX, minY, maxY])

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


def lukesAugment(image, vin, vout):
    from PIL import Image, ImageStat
    import numpy as np
    currentMean = np.mean(image)
    scale = (vout[1] - vout[0]) / (vin[1] - vin[0])

    image = image - currentMean
    image = image*scale
    image = image + currentMean*scale
    image = image - vin[0] + vout[0]
    image = Image.fromarray((image))
    return image