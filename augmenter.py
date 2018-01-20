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
from skimage import transform as tf

centrePerImage = False
patientList = ['PS']#, 'RR', 'DC', 'NS', 'PB']
augmentedList = [True]#, True, True, False, True]
augNumList = [5]#, 5, 5, 1, 5]
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

    for i in range(augNum):
        print('Doing augmentation ' + str(i))

        try:
            shutil.rmtree(tmpFolder)
        except:
            pass

        while (os.path.exists(tmpFolder)):
            9 - 7
        os.mkdir(tmpFolder)
        os.mkdir(tmpDicomDir)
        os.mkdir(tmpOuterBinaryDir)
        os.mkdir(tmpInnerBinaryDir)

        # Shear adjustment parameters
        IshearVal = 0.7 * randomVals[-1]
        # Shear adjustment
        afine_tf = tf.AffineTransform(shear=IshearVal)
        randomVals = np.delete(randomVals, -1)

        trueFileNum = 0

        innerFileList = sorted(os.listdir(innerBinaryReadDir))
        outerFileList = sorted(os.listdir(outerBinaryReadDir))
        dicomFileList = sorted(os.listdir(dicomReadDir))
        for innerBinaryFilename, outerBinaryFilename, dicomFilename in zip(innerFileList, outerFileList, dicomFileList):

            dicomFilepath = dicomReadDir + dicomFilename
            outerBinaryFilepath = outerBinaryReadDir + outerBinaryFilename
            innerBinaryFilepath = innerBinaryReadDir + innerBinaryFilename

            # Read the binaries
            innerBinaryImage = Image.open(innerBinaryFilepath)
            outerBinaryImage = Image.open(outerBinaryFilepath)
            innerBinaryImage = misc.imread(innerBinaryFilepath)
            outerBinaryImage = misc.imread(outerBinaryFilepath)

            trueFileNum = trueFileNum + 1

            #Read the dicom into a png
            dicomImage = dicom.read_file(dicomFilepath).pixel_array
            misc.imsave(tmpFolder + 'dicomTemp.png', dicomImage)
            dicomImage = misc.imread(tmpFolder + 'dicomTemp.png')
            os.remove(tmpFolder + 'dicomTemp.png')

            if augmented == True:
                # Contrast adjustment
                dicomImage = lukesAugment(dicomImage)

                # Apply transform to image data
                dicomImage = np.round(tf.warp(dicomImage/255, inverse_map=afine_tf)*255)
                innerBinaryImage = np.round(tf.warp(innerBinaryImage / 255, inverse_map=afine_tf) * 255)
                outerBinaryImage = np.round(tf.warp(outerBinaryImage / 255, inverse_map=afine_tf) * 255)

                # You could rotate image or scale image here
            else:
                dicomImage = lukesAugment(dicomImage)

            # Generates the names for the image file
            if augmented == True:
                innerBinaryWritename = 'Augment' + '%.2d' % (i+1) + 'InnerBinary' + '%.3d' % trueFileNum + 'Patient' + PatientID + '.png'
                outerBinaryWritename = 'Augment' + '%.2d' % (i+1) + 'OuterBinary' + '%.3d' % trueFileNum + 'Patient' + PatientID + '.png'
                dicomWritename = 'Augment' + '%.2d' % (i+1) + 'Original' + '%.3d' % trueFileNum + 'Patient' + PatientID + '.png'
            elif augmented == False:
                innerBinaryWritename = 'NonAugment' + 'InnerBinary' + '%.3d' % trueFileNum + 'Patient' + PatientID + '.png'
                outerBinaryWritename = 'NonAugment' + 'OuterBinary' + '%.3d' % trueFileNum + 'Patient' + PatientID + '.png'
                dicomWritename = 'NonAugment' + 'Original' + '%.3d' % trueFileNum + 'Patient' + PatientID + '.png'

            if centrePerImage:
                # Gets the bounding box for this particular image
                bBox = getFolderBoundingBox(tmpOuterBinaryDir)
                if bBox[1] - bBox[0] > boxSize or bBox[3] - bBox[2] > boxSize:
                    sys.exit('Houston we have a problem, the image isnt going to fit in ' + str(boxSize))
                xLower = int(bBox[0] - round((boxSize - bBox[1] + bBox[0]) / 2) if bBox[0] - round(
                    (boxSize - bBox[1] + bBox[0]) / 2) > 0 else 0)
                xUpper = xLower + boxSize
                yLower = int(bBox[2] - round((boxSize - bBox[3] + bBox[2]) / 2) if bBox[2] - round(
                    (boxSize - bBox[3] + bBox[2]) / 2) > 0 else 0)
                yUpper = yLower + boxSize

                if np.max(innerBinaryImage) < 10 or np.max(outerBinaryImage) < 10:
                    # This is so that you dont get an image unless both of them are defined
                    print('Blanking this one')
                    misc.imsave(innerBinaryWriteDir + innerBinaryWritename, np.zeros([boxSize, boxSize]))
                    misc.imsave(outerBinaryWriteDir + outerBinaryWritename, np.zeros([boxSize, boxSize]))
                else:
                    innerBinaryImage = innerBinaryImage[yLower:yUpper, xLower:xUpper]
                    if not innerBinaryImage.shape[0] == boxSize or not innerBinaryImage.shape[1] == boxSize:
                        sys.exit('Your shear is too large breh!')
                    misc.imsave(innerBinaryWriteDir + innerBinaryWritename, innerBinaryImage)
                    outerBinaryImage = outerBinaryImage[yLower:yUpper, xLower:xUpper]
                    if not outerBinaryImage.shape[0] == boxSize or not outerBinaryImage.shape[1] == boxSize:
                        sys.exit('Your shear is too large breh!')
                    misc.imsave(outerBinaryWriteDir + outerBinaryWritename, (outerBinaryImage-innerBinaryImage).clip(min=0))
                dicomImage = dicomImage[yLower:yUpper, xLower:xUpper]
                if not dicomImage.shape[0] == boxSize or not dicomImage.shape[1] == boxSize:
                    sys.exit('Your shear is too large breh!')
                misc.imsave(dicomWriteDir + dicomWritename, dicomImage)
            else:
                if np.max(innerBinaryImage) < 10 or np.max(outerBinaryImage) < 10:
                    # This is so that you dont get an image unless both of them are defined
                    print('Blanking this one')
                    misc.imsave(tmpInnerBinaryDir + innerBinaryWritename, np.zeros([512, 512]))
                    misc.imsave(tmpOuterBinaryDir + outerBinaryWritename, np.zeros([512, 512]))
                    misc.imsave(tmpDicomDir + dicomWritename, dicomImage)
                else:
                    misc.imsave(tmpOuterBinaryDir + outerBinaryWritename, outerBinaryImage)
                    misc.imsave(tmpInnerBinaryDir + innerBinaryWritename, innerBinaryImage)
                    misc.imsave(tmpDicomDir + dicomWritename, dicomImage)

        # Bases the chopping on the centre of the folder, reads in the images from the tmp folder and chops them
        if not centrePerImage:
            bBox = getFolderBoundingBox(tmpOuterBinaryDir)
            if bBox[1] - bBox[0] > boxSize or bBox[3] - bBox[2] > boxSize:
                sys.exit('Houston we have a problem, the image isnt going to fit in ' + str(boxSize))
            xLower = int(bBox[0] - round((boxSize - bBox[1] + bBox[0]) / 2) if bBox[0] - round(
                (boxSize - bBox[1] + bBox[0]) / 2) > 0 else sys.exit('Your skew is too big breh'))
            xUpper = xLower + boxSize
            yLower = int(bBox[2] - round((boxSize - bBox[3] + bBox[2]) / 2) if bBox[2] - round(
                (boxSize - bBox[3] + bBox[2]) / 2) > 0 else sys.exit('Your skew is too big breh'))
            yUpper = yLower + boxSize

            innerFileList = sorted(os.listdir(tmpInnerBinaryDir))
            outerFileList = sorted(os.listdir(tmpOuterBinaryDir))
            dicomFileList = sorted(os.listdir(tmpDicomDir))
            trueFileNum = 0
            for innerBinaryFilename, outerBinaryFilename, dicomFilename in zip(innerFileList, outerFileList, dicomFileList):
                trueFileNum = trueFileNum + 1
                if augmented == True:
                    innerBinaryWritename = 'Augment' + '%.2d' % (i + 1) + 'InnerBinary' + '%.3d' % trueFileNum + 'Patient' + PatientID + '.png'
                    outerBinaryWritename = 'Augment' + '%.2d' % (i + 1) + 'OuterBinary' + '%.3d' % trueFileNum + 'Patient' + PatientID + '.png'
                    dicomWritename = 'Augment' + '%.2d' % (i + 1) + 'Original' + '%.3d' % trueFileNum + 'Patient' + PatientID + '.png'
                elif augmented == False:
                    innerBinaryWritename = 'NonAugment' + 'InnerBinary' + '%.3d' % trueFileNum + 'Patient' + PatientID + '.png'
                    outerBinaryWritename = 'NonAugment' + 'OuterBinary' + '%.3d' % trueFileNum + 'Patient' + PatientID + '.png'
                    dicomWritename = 'NonAugment' + 'Original' + '%.3d' % trueFileNum + 'Patient' + PatientID + '.png'

                innerBinaryImage = misc.imread(tmpInnerBinaryDir + innerBinaryFilename)
                innerBinaryImage = innerBinaryImage[yLower:yUpper, xLower:xUpper]
                misc.imsave(innerBinaryWriteDir + innerBinaryFilename, innerBinaryImage)
                outerBinaryImage = misc.imread(tmpOuterBinaryDir + outerBinaryFilename)
                outerBinaryImage = outerBinaryImage[yLower:yUpper, xLower:xUpper]
                misc.imsave(outerBinaryWriteDir + outerBinaryFilename, (outerBinaryImage - innerBinaryImage).clip(min=0))
                dicomImage = misc.imread(tmpDicomDir + dicomFilename)
                dicomImage = dicomImage[yLower:yUpper, xLower:xUpper]
                misc.imsave(dicomWriteDir + dicomFilename, dicomImage)