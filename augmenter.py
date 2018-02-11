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
patientList = ['PB']
segmenter = 'Luke'
augmentedList = [True]
#augNumList = [10, 10, 1]
augNumList = [5]
tmpFolder = 'C:\\Users\\Luke\\Documents\\sharedFolder\\4YP\\4YP_Python\\tmp\\'
if not os.path.exists(tmpFolder):
            os.mkdir(tmpFolder)
boxSize = 256

for iteration in range(len(patientList)):

    PatientID = patientList[iteration]
    augmented = augmentedList[iteration]
    augNum = augNumList[iteration]

    #bulgeLocations = ['Center', 'Top', 'Bottom', 'Left', 'Right', 'Center', 'Top', 'Bottom', 'Left', 'Right']
    #bulgeDirections = [-1, -1, -1, -1, -1, 1, 1, 1, 1, 1]
    #shearValues = np.concatenate((np.linspace(-0.35, 0.35, np.floor(augNum/2)), np.linspace(-0.35, 0.35, np.ceil(augNum/2))))
    shearValues = np.linspace(-0.35, 0.35, augNum)

    print('Augmenting patient ' + PatientID)

    # Set read and write directories for the images
    preAugmentationRootDir = 'C:\\Users\\Luke\\Documents\\sharedFolder\\4YP\\Images\\' + segmenter + '_' + PatientID + '\\preAugmentation\\'
    postAugmentationRootDir = 'C:\\Users\\Luke\\Documents\\sharedFolder\\4YP\\Images\\' + segmenter + '_' + PatientID + '\\postAugmentation\\'
    innerBinaryReadDir = preAugmentationRootDir + 'innerBinary\\'
    innerBinaryWriteDir = postAugmentationRootDir + '\\innerAugmented\\'
    outerBinaryReadDir = preAugmentationRootDir + 'outerBinary\\'
    outerBinaryWriteDir = postAugmentationRootDir + 'outerAugmented\\'
    dicomReadDir = preAugmentationRootDir + 'dicoms\\'
    dicomWriteDir = postAugmentationRootDir + 'croppedDicoms\\'

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

    #for i, bulgeLocation, bulgeDirection, shearCoefficient in zip(np.linspace(0, augNum-1, augNum, dtype='int'), bulgeLocations, bulgeDirections, shearValues):
    for i, shearCoefficient in zip(np.linspace(0, augNum - 1, augNum, dtype='int'), shearValues):
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

        # Shear adjustment
        afine_tf = tf.AffineTransform(shear=shearCoefficient)

        trueFileNum = 0

        innerFileList = sorted(os.listdir(innerBinaryReadDir))
        outerFileList = sorted(os.listdir(outerBinaryReadDir))
        dicomFileList = sorted(os.listdir(dicomReadDir))
        for innerBinaryFilename, outerBinaryFilename, dicomFilename in zip(innerFileList, outerFileList, dicomFileList):

            trueFileNum = trueFileNum + 1

            dicomFilepath = dicomReadDir + dicomFilename
            outerBinaryFilepath = outerBinaryReadDir + outerBinaryFilename
            innerBinaryFilepath = innerBinaryReadDir + innerBinaryFilename

            # Read the binaries
            innerBinaryImage = misc.imread(innerBinaryFilepath, flatten=True)
            outerBinaryImage = misc.imread(outerBinaryFilepath, flatten=True)

            #Read the dicom into a png
            dicomImage = dicom.read_file(dicomFilepath).pixel_array
            #misc.imsave(tmpFolder + 'dicomTemp.png', dicomImage)
            misc.toimage(255*(dicomImage / 4095), cmin=0.0, cmax=255).save(tmpFolder + 'dicomTemp.png')
            #if trueFileNum == 238:
            #    plt.imshow(255*(dicomImage / 4095), cmap='gray')
            #    plt.show()
            dicomImage = misc.imread(tmpFolder + 'dicomTemp.png', flatten=True)
            os.remove(tmpFolder + 'dicomTemp.png')
            if augmented == True:
                # Contrast adjustment
                dicomImage = lukesAugment(dicomImage)

                # Apply the divergence transformation
                #if trueFileNum > 50 and np.max(innerBinaryImage) == 255 and not isDoubleAAA(innerBinaryImage):
                #    scaler = (np.where(np.isin(outerBinaryImage + innerBinaryImage, 255))[0].size) / 200
                #    dicomImage = lukesImageDiverge(dicomImage, getImageEdgeCoordinates(innerBinaryImage, bulgeLocation), bulgeDirection*scaler)
                #    innerBinaryImage = lukesImageDiverge(innerBinaryImage, getImageEdgeCoordinates(innerBinaryImage, bulgeLocation), bulgeDirection * scaler)
                #    outerBinaryImage = lukesImageDiverge(outerBinaryImage, getImageEdgeCoordinates(innerBinaryImage, bulgeLocation), bulgeDirection * scaler)

                # Apply shear to images
                dicomImage = np.round(tf.warp(dicomImage/255, inverse_map=afine_tf)*255)
                innerBinaryImage = np.round(tf.warp(innerBinaryImage / 255, inverse_map=afine_tf) * 255)
                outerBinaryImage = np.round(tf.warp(outerBinaryImage / 255, inverse_map=afine_tf) * 255)

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
                    #Image.fromarray(dicomImage).convert('RGB').save(tmpDicomDir + dicomWritename)
                    misc.imsave(tmpOuterBinaryDir + outerBinaryWritename, outerBinaryImage)
                    misc.imsave(tmpInnerBinaryDir + innerBinaryWritename, innerBinaryImage)
                    #misc.imsave(tmpDicomDir + dicomWritename, dicomImage)
                    misc.toimage(dicomImage, cmin=0.0, cmax=255).save(tmpDicomDir + dicomWritename)

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

                innerBinaryImage = misc.imread(tmpInnerBinaryDir + innerBinaryFilename, flatten=True)
                innerBinaryImage = innerBinaryImage[yLower:yUpper, xLower:xUpper]
                misc.imsave(innerBinaryWriteDir + innerBinaryFilename, innerBinaryImage)
                outerBinaryImage = misc.imread(tmpOuterBinaryDir + outerBinaryFilename, flatten=True)
                outerBinaryImage = outerBinaryImage[yLower:yUpper, xLower:xUpper]
                misc.imsave(outerBinaryWriteDir + outerBinaryFilename, lukesBinarize(outerBinaryImage - innerBinaryImage))
                dicomImage = misc.imread(tmpDicomDir + dicomFilename, flatten=True)
                dicomImage = dicomImage[yLower:yUpper, xLower:xUpper]
                misc.toimage(dicomImage, cmin=0.0, cmax=255).save(dicomWriteDir + dicomFilename)
                #misc.imsave(dicomWriteDir + dicomFilename, dicomImage)