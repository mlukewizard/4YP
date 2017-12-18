from __future__ import division
from random import *
import numpy as np
from imageProcessingFuntions import *
import os, shutil
import scipy
import scipy.misc as misc
import dicom
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
from scipy.ndimage.interpolation import affine_transform
from skimage import io
from skimage import transform as tf
import sys

tmpFolder = '/media/sf_sharedFolder/4YP/4YP_Python/tmp/'

#Set read and write directories for the images
innerBinaryReadDir = '/media/sf_sharedFolder/4YP/Images/Regent_RR/preAugmentation/innerBinary/'
innerBinaryWriteDir = '/media/sf_sharedFolder/4YP/Images/Regent_RR/postAugmentation/innerAugmented/'
outerBinaryReadDir = '/media/sf_sharedFolder/4YP/Images/Regent_RR/preAugmentation/outerBinary/'
outerBinaryWriteDir = '/media/sf_sharedFolder/4YP/Images/Regent_RR/postAugmentation/outerAugmented/'
dicomReadDir = '/media/sf_sharedFolder/4YP/Images/Regent_RR/preAugmentation/dicoms/'
dicomWriteDir = '/media/sf_sharedFolder/4YP/Images/Regent_RR/postAugmentation/croppedDicoms/'

PatientID = 'RR'
augmented = False
augNum = 1
randomVals = np.linspace(-0.5, 0.5, augNum)

tmpDicomDir = tmpFolder + 'temporaryDicoms/'
tmpOuterBinaryDir = tmpFolder + 'temporaryOuterBinaries/'
tmpInnerBinaryDir = tmpFolder + 'temporaryInnerBinaries/'


counter = 0
for i in range(augNum):

    try:
        shutil.rmtree(tmpFolder)
    except:
        pass

    while (os.path.exists(tmpFolder)):
        9-7
    os.mkdir(tmpFolder)
    os.mkdir(tmpDicomDir)
    os.mkdir(tmpOuterBinaryDir)
    os.mkdir(tmpInnerBinaryDir)

    #contrast adjustment parameters
    upper = 0
    while upper < 210:
        upper = 255*random()

    # Shear adjustment parameters

    index = (np.abs(randomVals + uniform(-0.625, 0.625))).argmin()
    print str(index)
    IshearVal = 0.6 * randomVals[index]
    randomVals = np.delete(randomVals, index)

    trueFileNum = 0

    fileList = sorted(os.listdir(innerBinaryReadDir))
    fileList = filter(lambda k: '130' in k, fileList)

    for filename in fileList:

        dicomImage = np.ndarray([512, 512])

        innerBinaryFilename = filename
        innerBinaryFilepath = innerBinaryReadDir + innerBinaryFilename

        trueFileNum = trueFileNum + 1
        newFileName = filename
        splitted = newFileName.split('Binary.png')
        firstHalf = splitted[0]
        splitted2 = firstHalf.split('inner')
        imageNumber = splitted2[1]

        dicomFilename = 'IMG00' + imageNumber
        dicomFilepath = dicomReadDir + dicomFilename

        outerBinaryFilename = 'outer' + imageNumber + 'Binary.png'
        outerBinaryFilepath = outerBinaryReadDir + outerBinaryFilename

        # Read the binaries
        innerBinaryImage = Image.open(innerBinaryFilepath)
        outerBinaryImage = Image.open(outerBinaryFilepath)

        #Read the dicom into a png
        inputDicomImage = dicom.read_file(dicomFilepath)
        dicomImage[:, :] = inputDicomImage.pixel_array
        misc.imsave(tmpFolder + 'dicomTemp.png', dicomImage)
        dicomImage = misc.imread(tmpFolder + 'dicomTemp.png')
        dicomImage = Image.fromarray((dicomImage))
        os.remove(tmpFolder + 'dicomTemp.png')

        currentLower = np.amin(dicomImage)
        currentUpper = np.amax(dicomImage)

        if augmented == True:
            # Contrast adjustment
            dicomImage = lukesAugment(dicomImage)
            #Shear adjustment
            afine_tf = tf.AffineTransform(shear=IshearVal)

            # Apply transform to image data
            dicomImage = np.round(tf.warp(np.array(dicomImage)/255, inverse_map=afine_tf)*255)
            innerBinaryImage = np.round(tf.warp(np.array(innerBinaryImage) / 255, inverse_map=afine_tf) * 255)
            outerBinaryImage = np.round(tf.warp(np.array(outerBinaryImage) / 255, inverse_map=afine_tf) * 255)

            dicomImage = Image.fromarray((dicomImage))
            innerBinaryImage = Image.fromarray((innerBinaryImage))
            outerBinaryImage = Image.fromarray((outerBinaryImage))

            # You could rotate image or scale image

        else:
            image = np.array(dicomImage)
            plt.imshow(image, cmap='gray')
            plt.show()
            dicomImage = lukesAugment(dicomImage)
            dicomImage.show()
        if augmented == True:
            innerBinaryWritename = 'Augment' + '%.2d' % (i+1) + 'InnerBinary' + '%.3d' % trueFileNum + 'Patient' + PatientID + '.png'
            outerBinaryWritename = 'Augment' + '%.2d' % (i+1) + 'OuterBinary' + '%.3d' % trueFileNum + 'Patient' + PatientID + '.png'
            dicomWritename = 'Augment' + '%.2d' % (i+1) + 'Original' + '%.3d' % trueFileNum + 'Patient' + PatientID + '.png'
        elif augmented == False:
            innerBinaryWritename = 'NonAugment' + 'InnerBinary' + '%.3d' % trueFileNum + 'Patient' + PatientID + '.png'
            outerBinaryWritename = 'NonAugment' + 'OuterBinary' + '%.3d' % trueFileNum + 'Patient' + PatientID + '.png'
            dicomWritename = 'NonAugment' + 'Original' + '%.3d' % trueFileNum + 'Patient' + PatientID + '.png'

        innerBinaryImage.convert('RGB').save(tmpInnerBinaryDir + innerBinaryWritename,'PNG')
        outerBinaryImage.convert('RGB').save(tmpOuterBinaryDir + outerBinaryWritename,'PNG')
        dicomImage.convert('RGB').save(tmpDicomDir + dicomWritename,'PNG')

    bBox = getFolderBoundingBox(tmpOuterBinaryDir)
    if bBox[1]-bBox[0] > 256 or bBox[3]-bBox[2] > 256:
        sys.exit('Houston we have a problem, the image isnt going to fit in 256x256')

    xLower = int(bBox[0] - round((256 - bBox[1]+bBox[0])/2) if bBox[0] - round((256 - bBox[1]+bBox[0])/2) > 0 else 0)
    xUpper = xLower + 256
    yLower = int(bBox[2] - round((256 - bBox[3]+bBox[2])/2) if bBox[2] - round((256 - bBox[3]+bBox[2])/2) > 0 else 0)
    yUpper = yLower + 256

    fileList = sorted(os.listdir(tmpOuterBinaryDir))
    for filename in fileList:
        image = Image.open(tmpOuterBinaryDir + filename)
        image = np.array(image)[yLower:yUpper, xLower:xUpper]
        image = Image.fromarray((image))
        image.convert('RGB').save(outerBinaryWriteDir + filename, 'PNG')

    fileList = sorted(os.listdir(tmpInnerBinaryDir))
    for filename in fileList:
        image = Image.open(tmpInnerBinaryDir + filename)
        image = np.array(image)[yLower:yUpper, xLower:xUpper]
        image = Image.fromarray((image))
        image.convert('RGB').save(innerBinaryWriteDir + filename, 'PNG')

    fileList = sorted(os.listdir(tmpDicomDir))
    for filename in fileList:
        image = Image.open(tmpDicomDir + filename)
        image = np.array(image)[yLower:yUpper, xLower:xUpper]
        image = Image.fromarray((image))
        image.convert('RGB').save(dicomWriteDir + filename, 'PNG')
