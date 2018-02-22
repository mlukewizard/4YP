from __future__ import print_function
from myFunctions import *
import dicom
import shutil
import os
import time
import sys
import math
from keras.callbacks import ModelCheckpoint
from keras.models import Model, load_model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from keras.optimizers import Adam
from keras import losses
import numpy as np
import h5py
import matplotlib
import matplotlib.pyplot as plt
from scipy import ndimage
import scipy.misc as misc
from PIL import Image, ImageEnhance
from uuid import getnode as get_mac
from keras import backend as K
def my_loss(y_true, y_pred):
    return K.mean(K.binary_crossentropy(y_true[:, 2, :, :, :], y_pred[:, 2, :, :, :]))
losses.my_loss = my_loss

def main():

    indexStartLocations = {}
    boxSize = 256

    tmpFolder = 'C:/Users/Luke/Documents/sharedFolder/4YP/4YP_Python/tmp/'
    model_file = 'C:/Users/Luke/Documents/sharedFolder/4YP\Models/12thFeb/weights.15-0.01.h5'
    tmpDicomDir = 'E:/dicomFolders/'
    tmpPredictionDir = 'E:/predictionFolders/'
    bankDicomDir = 'D:/allCases/'
    bankPredictionDir = 'D:/processedCases/'

    # Loads the model
    model = load_model(model_file)
    patientsWeHaveSegmented = [x[0:2] for x in os.listdir(bankPredictionDir)]
    patientsToSegmentList = sorted([x for x in os.listdir(bankDicomDir) if x[0:2] not in patientsWeHaveSegmented])

    for patientNum, specificEntry in enumerate(patientsToSegmentList):
        print('Working on patient ' + str(patientNum) +'/'+str(len(patientsToSegmentList)))
        patientID = specificEntry[0:2]
        indexStartLocation = indexStartLocations[patientID] if patientID in indexStartLocations.keys() else 0
        doPatientSegmentation(specificEntry, patientsToSegmentList, indexStartLocation, model, boxSize, tmpFolder, tmpDicomDir, tmpPredictionDir, bankDicomDir, bankPredictionDir)


def cleanPointCloud(innerPC, outerPC, patientID, predictionFolder):
    # Making the outer point cloud solid
    outerPC = outerPC + innerPC
    outerPredictionFolder = predictionFolder + 'outerPredictions/'
    innerPredictionFolder = predictionFolder + 'innerPredictions/'

    print('Cleaning up the inner point cloud')
    # Cleans up the inner point cloud
    innerPC = np.where(innerPC > 140, 255, 0)
    innerPC = getMaxConnectedComponent(innerPC)
    for i in range(innerPC.shape[0]):
        newPC, num_features = ndimage.label(innerPC[i, :, :])
        if num_features > 2:
            labelCounts = []
            for j in range(num_features):
                labelCounts.append(len(np.where(innerPC[i, :, :] == j)[0]))
            maxIndex1 = labelCounts.index(sorted(labelCounts)[-2])
            maxIndex2 = labelCounts.index(sorted(labelCounts)[-3])
            innerPC[i, :, :] = np.where(innerPC[i, :, :] == maxIndex1, 255, 0) + np.where(innerPC[i, :, :] == maxIndex2, 255, 0)
        if np.count_nonzero(innerPC[i, :, :]) > 80:
            innerPC[i, :, :] = ndimage.binary_fill_holes(innerPC[i, :, :])
            innerPC[i, :, :] = ndimage.binary_closing(innerPC[i, :, :], iterations=2)
        else:
            innerPC[i, :, :] = np.zeros([512, 512])

    print('Cleaning up the outer point cloud')
    outerPC = np.where(outerPC > 60, 255, 0)
    outerPC = getMaxConnectedComponent(outerPC)
    for i in range(outerPC.shape[0]):
        if np.count_nonzero(innerPC[i, :, :]) > 0:
            # If theres more than 2 features, then just get the biggest two
            newPC, num_features = ndimage.label(outerPC[i, :, :])
            if num_features > 2:
                # Gets the largest two components
                labelCounts = []
                for j in range(num_features):
                    labelCounts.append(len(np.where(outerPC[i, :, :] == j)[0]))
                maxIndex1 = labelCounts.index(sorted(labelCounts)[-2])
                maxIndex2 = labelCounts.index(sorted(labelCounts)[-3])
                outerPC[i, :, :] = np.where(outerPC[i, :, :] == maxIndex1, 255, 0) + np.where(outerPC[i, :, :] == maxIndex2, 255, 0)

            outerPC[i, :, :] = ndimage.binary_fill_holes(outerPC[i, :, :])
            outerPC[i, :, :] = ndimage.binary_closing(outerPC[i, :, :], iterations=4)
        else:
            outerPC[i, :, :] = np.zeros([512, 512])

    print('Writing Inner Point Cloud')
    np.save(predictionFolder + patientID + 'ThickInnerPointCloud' + '.npy', innerPC)
    print('Writing Outer Point Cloud')
    np.save(predictionFolder + patientID + 'ThickOuterPointCloud' + '.npy', outerPC)

    print('Writing cleaned images')
    for i in range(outerPC.shape[0]):
        if np.max(np.max(outerPC[i, :, :])) > 0:
            outerPC[i, :, :] = outerPC[i, :, :] / np.max(np.max(outerPC[i, :, :]))
        misc.toimage(outerPC[i, :, :], cmin=0.0, cmax=1).save((outerPredictionFolder + 'outerPredictedCleaned_' + "%03d" % (i,)).split('.dcm')[0] + '.png')
    for i in range(innerPC.shape[0]):
        if np.max(np.max(innerPC[i, :, :])) > 0:
            innerPC[i, :, :] = innerPC[i, :, :] / np.max(np.max(outerPC[i, :, :]))
        misc.toimage(innerPC[i, :, :], cmin=0.0, cmax=1).save((innerPredictionFolder + 'innerPredictedCleaned_' + "%03d" % (i,)).split('.dcm')[0] + '.png')

def doPatientSegmentation(specificEntry, uncompletedFileList, indexStartLocation, model, boxSize, tmpFolder, tmpDicomDir, tmpPredictionDir, bankDicomDir, bankPredictionDir):
    patientID = specificEntry[0:2]

    tryUpdateFileSystem(specificEntry, uncompletedFileList, tmpDicomDir, tmpPredictionDir, bankDicomDir, bankPredictionDir)

    predictionFolder = tmpPredictionDir + patientID + '_predictions/'
    outerPredictionFolder = predictionFolder + 'outerPredictions/'
    innerPredictionFolder = predictionFolder + 'innerPredictions/'
    [os.mkdir(myFolder) for myFolder in [tmpFolder, predictionFolder, innerPredictionFolder, outerPredictionFolder] if not os.path.exists(myFolder)]

    dicomList = sorted(os.listdir(tmpDicomDir + specificEntry))
    print('You should check the following are in alphabetical/numerical order')
    print(dicomList[0])
    print(dicomList[1])
    print(dicomList[2])

    loopTimes = [12]

    if not all(os.path.exists(myFolder) for myFolder in [predictionFolder + patientID + '_innerBinaryArray' + '.npy',
                                                         predictionFolder + patientID + '_outerBinaryArray' + '.npy']):
        print('No previously made numpy arrays found')

        # Initialises the pointCloud
        innerPC = np.zeros([len(dicomList), 512, 512])
        outerPC = np.zeros([len(dicomList), 512, 512])


        for loopCount, k in enumerate(range(indexStartLocation, len(dicomList))):
            loopStartTime = time.time()
            # Predicts the location of the aneurysm
            secondsRemaining = (len(dicomList) - k) * (np.median(loopTimes)) / (loopCount + 1)
            print("Predicting slice " + str(k) + '/' + str(len(dicomList)) + " for patient " + patientID + ". Estimated time remaining: " + str(int(np.floor(secondsRemaining / 60))) + " minutes and " + str(int(((secondsRemaining / 60) - np.floor(secondsRemaining / 60)) * 60)) + " seconds")

            # Constructing a suitable array to feed to the algorithm so it can segment the slice in question
            modelInputArray = np.expand_dims(ConstructArraySlice(dicomList, tmpDicomDir, k, boxSize, tmpFolder, centralLocation=centralCoordinate), axis=0)

            # Uses the algorithm to predict the location of the aneurysm
            output = model.predict(modelInputArray) * 255

            # Gets the location of the 256x256 box in the 512x512 image
            upperRow = int(centralCoordinate[0] - round(boxSize / 2))
            lowerRow = int(upperRow + boxSize)
            leftColumn = int(centralCoordinate[1] - round(boxSize / 2))
            rightColumn = int(leftColumn + boxSize)

            resizedOuterImage = np.zeros([512, 512])
            resizedOuterImage[upperRow:lowerRow, leftColumn:rightColumn] = output[0, 2, :, :, 1]
            # misc.toimage(resizedOuterImage, cmin=0.0, cmax=255).save((algoOuterFolder + 'outerPredicted_' + dicomList[k]).split('.dcm')[0] + '.png')
            resizedInnerImage = np.zeros([512, 512])
            resizedInnerImage[upperRow:lowerRow, leftColumn:rightColumn] = output[0, 2, :, :, 0]
            # misc.toimage(resizedInnerImage, cmin=0.0, cmax=255).save((algoInnerFolder + 'innerPredicted_' + dicomList[k]).split('.dcm')[0] + '.png')

            outerPC[k, :, :] = resizedOuterImage
            innerPC[k, :, :] = resizedInnerImage

            # Updates the location of the centre of mass for the next iteration
            newLocation = ndimage.measurements.center_of_mass(resizedInnerImage + resizedOuterImage)
            centralCoordinate = [int(centralCoordinate[0] + (newLocation[0] - centralCoordinate[0]) / np.power(loopCount + 1, 0.2)), int(centralCoordinate[1] + (newLocation[1] - centralCoordinate[1]) / np.power(loopCount + 1, 0.2))]
            centralCoordinate[1] = clamp(centralCoordinate[1], 192, 320)
            centralCoordinate[0] = clamp(centralCoordinate[0], 192, 320)

            loopTimes.append(time.time() - loopStartTime)

        np.save(predictionFolder + patientID + '_outerBinaryArray' + '.npy', outerPC)
        np.save(predictionFolder + patientID + '_innerBinaryArray' + '.npy', innerPC)
    else:
        print('Using previously constructed numpy arrays')
        innerPC = np.load(predictionFolder + patientID + '_innerBinaryArray' + '.npy')
        outerPC = np.load(predictionFolder + patientID + '_outerBinaryArray' + '.npy')

    cleanPointCloud(innerPC, outerPC, patientID, predictionFolder)

def tryUpdateFileSystem(specificEntry, uncompletedFileList, tmpDicomDir, tmpPredictionDir, bankDicomDir, bankPredictionDir):
    currentIndex = uncompletedFileList.index(specificEntry)
    dicomFoldersWeWant = {uncompletedFileList[currentIndex % len(uncompletedFileList)],
                   uncompletedFileList[currentIndex +1 % len(uncompletedFileList)],
                   uncompletedFileList[currentIndex +2 % len(uncompletedFileList)]}
    dicomFoldersWeHave = set(os.listdir(tmpDicomDir))
    if dicomFoldersWeWant == dicomFoldersWeHave:
        print('File system updated')
        return True
    else:
        if not os.path.exists(bankDicomDir):
            return False
        else:
            print('Updating file system...')
            predictionFoldersWeWant = {uncompletedFileList[currentIndex % len(uncompletedFileList)][0:2] + '_processed',
                                        uncompletedFileList[currentIndex + 1 % len(uncompletedFileList)][0:2] + '_processed',
                                        uncompletedFileList[currentIndex + 2 % len(uncompletedFileList)][0:2] + '_processed'}
            predictionFoldersWeHave = set(os.listdir(tmpPredictionDir))
            for newlyMadePredictionFolder in list(predictionFoldersWeHave - predictionFoldersWeWant):
                shutil.move(tmpPredictionDir + newlyMadePredictionFolder, bankPredictionDir + newlyMadePredictionFolder)
            for previouslyUsedDicomFolder in list(dicomFoldersWeHave - dicomFoldersWeWant):
                shutil.rmtree(tmpDicomDir + previouslyUsedDicomFolder)
            for dicomFolderWeWillNeed in list(dicomFoldersWeWant - dicomFoldersWeHave):
                shutil.copytree(bankDicomDir + dicomFolderWeWillNeed, tmpDicomDir + dicomFolderWeWillNeed)

if __name__ == '__main__':
    main()