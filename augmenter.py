from __future__ import division
from random import *
import numpy as np
import sys
from myFunctions import *
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


patientList = ['NS', 'PB', 'PS', 'RR', 'DC']
augmentedList = [False, True, True, True, True]
augNumList = [1, 6, 6, 6, 6]
tmpFolder = 'C:\\Users\\Luke\\Documents\\sharedFolder\\4YP\\4YP_Python\\tmp\\'
if not os.path.exists(tmpFolder):
            os.mkdir(tmpFolder)
boxSize = 256

for iteration in range(len(patientList)):

    PatientID = patientList[iteration]
    augmented = augmentedList[iteration]
    augNum = augNumList[iteration]

    randomVals = np.linspace(-0.5, 0.5, augNum)

    print('Augmenting patient ' + PatientID)

    # Set read and write directories for the images
    preAugmentationRootDir = 'C:\\Users\\Luke\\Documents\\sharedFolder\\4YP\\Images\\Regent_' + PatientID + '\\preAugmentation\\'
    postAugmentationRootDir = 'C:\\Users\\Luke\\Documents\\sharedFolder\\4YP\\Images\\Regent_' + PatientID + '\\postAugmentation\\'
    innerBinaryReadDir = preAugmentationRootDir + 'innerBinary\\'
    innerBinaryWriteDir = postAugmentationRootDir + '\\innerAugmented\\'
    outerBinaryReadDir = preAugmentationRootDir + 'outerBinary\\'
    outerBinaryWriteDir = postAugmentationRootDir + 'outerAugmented\\'
    dicomReadDir = preAugmentationRootDir + 'dicoms\\'
    dicomWriteDir = postAugmentationRootDir + 'croppedDicoms\\'

    if not os.path.exists(preAugmentationRootDir):
        os.mkdir(preAugmentationRootDir)
    if not os.path.exists(postAugmentationRootDir):
        os.mkdir(postAugmentationRootDir)
    if not os.path.exists(innerBinaryWriteDir):
        os.mkdir(innerBinaryWriteDir)
    if not os.path.exists(outerBinaryWriteDir):
        os.mkdir(outerBinaryWriteDir)
    if not os.path.exists(dicomWriteDir):
        os.mkdir(dicomWriteDir)

    tmpDicomDir = tmpFolder + 'temporaryDicoms/'
    tmpOuterBinaryDir = tmpFolder + 'temporaryOuterBinaries/'
    tmpInnerBinaryDir = tmpFolder + 'temporaryInnerBinaries/'


    counter = 0
    for i in range(augNum):
        print('Doing augmentation ' + str(i))
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
        IshearVal = 0.6 * randomVals[index]
        randomVals = np.delete(randomVals, index)

        trueFileNum = 0

        fileList = sorted(os.listdir(innerBinaryReadDir))
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
                #plt.imshow(image, cmap='gray')
                #plt.show()
                dicomImage = lukesAugment(dicomImage)
                #dicomImage.show()
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
        if bBox[1]-bBox[0] > boxSize or bBox[3]-bBox[2] > boxSize:
            sys.exit('Houston we have a problem, the image isnt going to fit in ' + str(boxSize))

        xLower = int(bBox[0] - round((boxSize - bBox[1]+bBox[0])/2) if bBox[0] - round((boxSize - bBox[1]+bBox[0])/2) > 0 else 0)
        xUpper = xLower + boxSize
        yLower = int(bBox[2] - round((boxSize - bBox[3]+bBox[2])/2) if bBox[2] - round((boxSize - bBox[3]+bBox[2])/2) > 0 else 0)
        yUpper = yLower + boxSize

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
