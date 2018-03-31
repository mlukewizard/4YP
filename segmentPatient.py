from __future__ import print_function
from myFunctions import *
import dicom
import shutil
import os
import time
import gc
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
import subprocess
import matplotlib.pyplot as plt
from scipy import ndimage
import scipy.misc as misc
from PIL import Image, ImageEnhance
from uuid import getnode as get_mac
from keras import backend as K
def my_loss(y_true, y_pred):
    return K.mean(K.binary_crossentropy(y_true[:, 2, :, :, :], y_pred[:, 2, :, :, :]))
losses.my_loss = my_loss


def getCumulativeDistribution(pointCloud):
    distArray = [np.sum(pointCloud[0, :, :])]
    for i in range(1, pointCloud.shape[0]):
            distArray.append(distArray[i-1] + np.sum(pointCloud[i, :, :]))
    return np.array(distArray)/distArray[-1]

def getMaxConnectedComponent(PC):
    from scipy import ndimage
    import numpy as np
    newPC, num_features = ndimage.label(PC)
    if num_features > 2:
        labelCounts = {}
        for j in range(1, num_features + 1):
            labelCounts[j] = len(np.where(newPC == j)[0])
        maxIndex1 = [key for key, value in labelCounts.items() if value == max(labelCounts.values())][0]
        PC = np.where(newPC == maxIndex1, 255, 0)
    else:
        PC = np.where(newPC == 1, 255, 0)
    return PC

def get2MaxConnectedComponents(PC):
    from scipy import ndimage
    import numpy as np
    newPC, num_features = ndimage.label(PC)
    if num_features > 2:
        labelCounts = {}
        for j in range(1, num_features + 1):
            labelCounts[j] = len(np.where(newPC == j)[0])
        maxIndex1 = [key for key, value in labelCounts.items() if value == max(labelCounts.values())][0]
        del labelCounts[maxIndex1]
        maxIndex2 = [key for key, value in labelCounts.items() if value == max(labelCounts.values())][0]
        PC = np.where(newPC == maxIndex1, 255, 0) + np.where(newPC == maxIndex2, 255, 0)
    return PC

def makeOrPopulateFolders(innerPC, outerPC, patientID, predictionFolder):
    # Making the outer point cloud solid
    outerPC = outerPC + innerPC

    outerPredictionFolder = predictionFolder + 'outerPredictions/'
    innerPredictionFolder = predictionFolder + 'innerPredictions/'
    innerAlgoFolder = predictionFolder + 'algoInners/'
    outerAlgoFolder = predictionFolder + 'algoOuters/'

    if not os.path.exists(outerAlgoFolder) and not os.path.exists(innerAlgoFolder):
        [os.mkdir(myFolder) for myFolder in [innerAlgoFolder, outerAlgoFolder] if not os.path.exists(myFolder)]

        print('Inner and outer algo prediction folders havent been made prevous, populating these...')
        for i in range(outerPC.shape[0]):
            if np.max(np.max(outerPC[i, :, :])) > 0:
                mySlice = outerPC[i, :, :] / np.max(np.max(outerPC[i, :, :]))
            misc.toimage(mySlice, cmin=0.0, cmax=1).save((outerAlgoFolder + 'outerPredicted_' + "%03d" % (i,)).split('.dcm')[0] + '.png')

        for i in range(innerPC.shape[0]):
            if np.max(np.max(innerPC[i, :, :])) > 0:
                mySlice = innerPC[i, :, :] / np.max(np.max(innerPC[i, :, :]))
            misc.toimage(mySlice, cmin=0.0, cmax=1).save((innerAlgoFolder + 'innerPredicted_' + "%03d" % (i,)).split('.dcm')[0] + '.png')


    innerDist = getCumulativeDistribution(innerPC)

    return innerPC, outerPC, patientID, predictionFolder, innerDist

def cleanOuterPointCloud(outerPC, patientID, predictionFolder, innerDist):
    outerPredictionFolder = predictionFolder + 'outerPredictions/'

    print('Cleaning up the outer point cloud')
    for i in range(outerPC.shape[0]):
        outerSlice = np.where(outerPC[i, :, :] > 140, 255, 0)
        if np.sum(outerPC[i, :, :]) > 5000:
            if np.count_nonzero(outerSlice) < 100 and innerDist[i] < 0.8 and innerDist[i] > 0.2:
                outerSlice = np.where(outerPC[i, :, :] > 80, 255, 0)
                if np.count_nonzero(outerSlice) < 100:
                    outerSlice = np.where(outerPC[i, :, :] > 40, 255, 0)
                    if np.count_nonzero(outerSlice) < 100:
                        outerSlice = np.where(outerPC[i, :, :] > 20, 255, 0)
                        if np.count_nonzero(outerSlice) < 100:
                            outerSlice = np.where(outerPC[i, :, :] > 5, 255, 0)
                            if np.count_nonzero(outerSlice) < 100:
                                outerSlice = np.where(outerPC[i, :, :] > 2, 255, 0)
                                if np.count_nonzero(outerSlice) < 100:
                                    outerSlice = np.where(outerPC[i, :, :] > 0.4, 255, 0)
        if np.count_nonzero(outerSlice) >= 20:
            outerPC[i, :, :] = ndimage.binary_fill_holes(outerSlice)
            outerPC[i, :, :] = ndimage.binary_closing(outerPC[i, :, :], iterations=4)
        elif np.count_nonzero(outerSlice) < 20:
            outerPC[i, :, :] = np.zeros([512, 512])
    print('Getting max connected component')
    outerPC = getMaxConnectedComponent(outerPC)

    print('Writing cleaned outer images')
    for i in range(outerPC.shape[0]):
        if np.max(np.max(outerPC[i, :, :])) > 0:
            outerPC[i, :, :] = outerPC[i, :, :] / np.max(np.max(outerPC[i, :, :]))
        misc.toimage(outerPC[i, :, :], cmin=0.0, cmax=1).save((outerPredictionFolder + 'outerPredictedCleaned_' + "%03d" % (i,)).split('.dcm')[0] + '.png')

    print('Writing Outer Point Cloud')
    np.save(predictionFolder + patientID + 'ThickOuterPointCloud' + '.npy', outerPC)

def cleanInnerPointCloud(innerPC, patientID, predictionFolder, innerDist):

    innerPredictionFolder = predictionFolder + 'innerPredictions/'

    print('Cleaning up the inner point cloud')
    # Cleans up the inner point cloud
    for i in range(innerPC.shape[0]):
        innerSlice = np.where(innerPC[i, :, :] > 140, 255, 0)
        if np.sum(innerPC[i, :, :]) > 5000:
            if np.count_nonzero(innerSlice) < 100 and innerDist[i] < 0.8 and innerDist[i] > 0.2:
                innerSlice = np.where(innerPC[i, :, :] > 80, 255, 0)
                if np.count_nonzero(innerSlice) < 100:
                    innerSlice = np.where(innerPC[i, :, :] > 40, 255, 0)
                    if np.count_nonzero(innerSlice) < 100:
                        innerSlice = np.where(innerPC[i, :, :] > 20, 255, 0)
                        if np.count_nonzero(innerSlice) < 100:
                            innerSlice = np.where(innerPC[i, :, :] > 5, 255, 0)
                            if np.count_nonzero(innerSlice) < 100:
                                innerSlice = np.where(innerPC[i, :, :] > 2, 255, 0)
                                if np.count_nonzero(innerSlice) < 80:
                                    innerSlice = np.where(innerPC[i, :, :] > 0.4, 255, 0)
        if np.count_nonzero(innerSlice) >= 20:
            innerPC[i, :, :] = ndimage.binary_fill_holes(innerSlice)
            innerPC[i, :, :] = ndimage.binary_closing(innerPC[i, :, :], iterations=4)
        elif np.count_nonzero(innerSlice) < 20:
            innerPC[i, :, :] = np.zeros([512, 512])
    print('Getting max connected component')
    innerPC = getMaxConnectedComponent(innerPC)

    print('Writing cleaned inner images')
    for i in range(innerPC.shape[0]):
        if np.max(np.max(innerPC[i, :, :])) > 0:
            innerPC[i, :, :] = innerPC[i, :, :] / np.max(np.max(innerPC[i, :, :]))
        misc.toimage(innerPC[i, :, :], cmin=0.0, cmax=1).save((innerPredictionFolder + 'innerPredictedCleaned_' + "%03d" % (i,)).split('.dcm')[0] + '.png')

    print('Writing Inner Point Cloud')
    np.save(predictionFolder + patientID + 'ThickInnerPointCloud' + '.npy', innerPC)



def doPatientSegmentationWithoutStorage(specificEntry, patientsToSegmentList, indexStartLocation, model, boxSize, tmpFolder, bankDicomDir, bankPredictionDir):
    patientID = specificEntry[0:2]

    predictionFolder = bankPredictionDir + patientID + '_processed/'
    outerPredictionFolder = predictionFolder + 'outerPredictions/'
    innerPredictionFolder = predictionFolder + 'innerPredictions/'
    innerAlgoFolder = predictionFolder + 'algoInners/'
    outerAlgoFolder = predictionFolder + 'algoOuters/'
    [os.mkdir(myFolder) for myFolder in [tmpFolder, predictionFolder, innerPredictionFolder, outerPredictionFolder] if not os.path.exists(myFolder)]

    dicomList = sorted(os.listdir(bankDicomDir + specificEntry))
    print('You should check the following are in alphabetical/numerical order')
    print(dicomList[0])
    print(dicomList[1])
    print(dicomList[2])


    if not all(os.path.exists(myFolder) for myFolder in [predictionFolder + patientID + '_innerBinaryArray' + '.npy',
                                                         predictionFolder + patientID + '_outerBinaryArray' + '.npy']):
        print('No previously made numpy arrays found')

        [os.mkdir(myFolder) for myFolder in [innerAlgoFolder, outerAlgoFolder] if not os.path.exists(myFolder)]

        # Initialises the pointCloud
        innerPC = np.zeros([len(dicomList), 512, 512])
        outerPC = np.zeros([len(dicomList), 512, 512])

        centralCoordinate = [240, 256]

        loopStartTime = time.time()

        for loopCount, k in enumerate(range(indexStartLocation, len(dicomList))):

            # Predicts the location of the aneurysm
            secondsRemaining = (len(dicomList) - k) * (time.time() - loopStartTime) / (loopCount + 1)
            printString = "Predicting slice " + str(k+1) + '/' + str(len(dicomList)) + " for patient " + patientID + ". Estimated time remaining: " + str(int(np.floor(secondsRemaining / 60))) + " minutes and " + str(int(((secondsRemaining / 60) - np.floor(secondsRemaining / 60)) * 60)) + " seconds"
            print(printString)

            # Constructing a suitable array to feed to the algorithm so it can segment the slice in question
            modelInputArray = np.expand_dims(ConstructArraySlice(dicomList, bankDicomDir + specificEntry + '/', k, boxSize, tmpFolder, centralLocation=centralCoordinate), axis=0)

            # Uses the algorithm to predict the location of the aneurysm
            output = model.predict(modelInputArray) * 255

            # Gets the location of the 256x256 box in the 512x512 image
            upperRow = int(centralCoordinate[0] - round(boxSize / 2))
            lowerRow = int(upperRow + boxSize)
            leftColumn = int(centralCoordinate[1] - round(boxSize / 2))
            rightColumn = int(leftColumn + boxSize)

            resizedOuterImage = np.zeros([512, 512])
            resizedOuterImage[upperRow:lowerRow, leftColumn:rightColumn] = output[0, 2, :, :, 1]
            misc.toimage(resizedOuterImage, cmin=0.0, cmax=255).save((outerAlgoFolder + 'outerPredicted_' + dicomList[k]).split('.dcm')[0] + '.png')
            resizedInnerImage = np.zeros([512, 512])
            resizedInnerImage[upperRow:lowerRow, leftColumn:rightColumn] = output[0, 2, :, :, 0]
            misc.toimage(resizedInnerImage, cmin=0.0, cmax=255).save((innerAlgoFolder + 'innerPredicted_' + dicomList[k]).split('.dcm')[0] + '.png')

            outerPC[k, :, :] = resizedOuterImage
            innerPC[k, :, :] = resizedInnerImage

            # Updates the location of the centre of mass for the next iteration
            newLocation = ndimage.measurements.center_of_mass(resizedInnerImage + resizedOuterImage)
            centralCoordinate = [int(centralCoordinate[0] + (newLocation[0] - centralCoordinate[0]) / np.power(loopCount + 1, 0.2)), int(centralCoordinate[1] + (newLocation[1] - centralCoordinate[1]) / np.power(loopCount + 1, 0.2))]
            centralCoordinate[1] = clamp(centralCoordinate[1], 192, 320)
            centralCoordinate[0] = clamp(centralCoordinate[0], 192, 320)

        print('Writing outer predictions to numpy array')
        np.save(predictionFolder + patientID + '_outerBinaryArray' + '.npy', outerPC)
        print('Writing inner predictions to numpy array')
        np.save(predictionFolder + patientID + '_innerBinaryArray' + '.npy', innerPC)
    else:
        print('Using previously constructed numpy arrays')
        innerPC = np.load(predictionFolder + patientID + '_innerBinaryArray' + '.npy')
        outerPC = np.load(predictionFolder + patientID + '_outerBinaryArray' + '.npy')

    return innerPC, outerPC, patientID, predictionFolder

def doPatientSegmentationWithStorage(specificEntry, uncompletedFileList, indexStartLocation, model, boxSize, tmpFolder, tmpDicomDir, tmpPredictionDir, bankDicomDir, bankPredictionDir):
    patientID = specificEntry[0:2]

    tryUpdateFileSystem(specificEntry, uncompletedFileList, tmpDicomDir, tmpPredictionDir, bankDicomDir, bankPredictionDir)

    predictionFolder = tmpPredictionDir + patientID + '_processed/'
    outerPredictionFolder = predictionFolder + 'outerPredictions/'
    innerPredictionFolder = predictionFolder + 'innerPredictions/'
    innerAlgoFolder = predictionFolder + 'algoInners/'
    outerAlgoFolder = predictionFolder + 'algoOuters/'
    [os.mkdir(myFolder) for myFolder in [tmpFolder, predictionFolder, innerPredictionFolder, outerPredictionFolder] if not os.path.exists(myFolder)]

    dicomList = sorted(os.listdir(tmpDicomDir + specificEntry))
    print('You should check the following are in alphabetical/numerical order')
    print(dicomList[0])
    print(dicomList[1])
    print(dicomList[2])


    if not all(os.path.exists(myFolder) for myFolder in [predictionFolder + patientID + '_innerBinaryArray' + '.npy',
                                                         predictionFolder + patientID + '_outerBinaryArray' + '.npy']):
        print('No previously made numpy arrays found')

        [os.mkdir(myFolder) for myFolder in [innerAlgoFolder, outerAlgoFolder] if not os.path.exists(myFolder)]

        # Initialises the pointCloud
        innerPC = np.zeros([len(dicomList), 512, 512])
        outerPC = np.zeros([len(dicomList), 512, 512])

        centralCoordinate = [240, 256]

        fileSystemVerified = False
        loopStartTime = time.time()

        for loopCount, k in enumerate(range(indexStartLocation, len(dicomList))):


            if not fileSystemVerified:
                if tryUpdateFileSystem(specificEntry, uncompletedFileList, tmpDicomDir, tmpPredictionDir, bankDicomDir, bankPredictionDir) == True:
                    fileSystemVerified = True

            # Predicts the location of the aneurysm
            secondsRemaining = (len(dicomList) - k) * (time.time() - loopStartTime) / (loopCount + 1)
            if fileSystemVerified:
                printString = "Predicting slice " + str(k+1) + '/' + str(len(dicomList)) + " for patient " + patientID + ". Estimated time remaining: " + str(int(np.floor(secondsRemaining / 60))) + " minutes and " + str(int(((secondsRemaining / 60) - np.floor(secondsRemaining / 60)) * 60)) + " seconds"
            else:
                printString = "Warning, file system not up to date." + " Predicting slice " + str(k) + '/' + str(len(dicomList)) + " for patient " + patientID + ". Estimated time remaining: " + str(int(np.floor(secondsRemaining / 60))) + " minutes and " + str(int(((secondsRemaining / 60) - np.floor(secondsRemaining / 60)) * 60)) + " seconds"
            print(printString)

            # Constructing a suitable array to feed to the algorithm so it can segment the slice in question
            modelInputArray = np.expand_dims(ConstructArraySlice(dicomList, tmpDicomDir + specificEntry + '/', k, boxSize, tmpFolder, centralLocation=centralCoordinate), axis=0)

            # Uses the algorithm to predict the location of the aneurysm
            output = model.predict(modelInputArray) * 255

            # Gets the location of the 256x256 box in the 512x512 image
            upperRow = int(centralCoordinate[0] - round(boxSize / 2))
            lowerRow = int(upperRow + boxSize)
            leftColumn = int(centralCoordinate[1] - round(boxSize / 2))
            rightColumn = int(leftColumn + boxSize)

            resizedOuterImage = np.zeros([512, 512])
            resizedOuterImage[upperRow:lowerRow, leftColumn:rightColumn] = output[0, 2, :, :, 1]
            misc.toimage(resizedOuterImage, cmin=0.0, cmax=255).save((outerAlgoFolder + 'outerPredicted_' + dicomList[k]).split('.dcm')[0] + '.png')
            resizedInnerImage = np.zeros([512, 512])
            resizedInnerImage[upperRow:lowerRow, leftColumn:rightColumn] = output[0, 2, :, :, 0]
            misc.toimage(resizedInnerImage, cmin=0.0, cmax=255).save((innerAlgoFolder + 'innerPredicted_' + dicomList[k]).split('.dcm')[0] + '.png')

            outerPC[k, :, :] = resizedOuterImage
            innerPC[k, :, :] = resizedInnerImage

            # Updates the location of the centre of mass for the next iteration
            newLocation = ndimage.measurements.center_of_mass(resizedInnerImage + resizedOuterImage)
            centralCoordinate = [int(centralCoordinate[0] + (newLocation[0] - centralCoordinate[0]) / np.power(loopCount + 1, 0.2)), int(centralCoordinate[1] + (newLocation[1] - centralCoordinate[1]) / np.power(loopCount + 1, 0.2))]
            centralCoordinate[1] = clamp(centralCoordinate[1], 192, 320)
            centralCoordinate[0] = clamp(centralCoordinate[0], 192, 320)

        print('Writing outer predictions to numpy array')
        np.save(predictionFolder + patientID + '_outerBinaryArray' + '.npy', outerPC)
        print('Writing inner predictions to numpy array')
        np.save(predictionFolder + patientID + '_innerBinaryArray' + '.npy', innerPC)
    else:
        print('Using previously constructed numpy arrays')
        innerPC = np.load(predictionFolder + patientID + '_innerBinaryArray' + '.npy')
        outerPC = np.load(predictionFolder + patientID + '_outerBinaryArray' + '.npy')

def tryUpdateFileSystem(specificEntry, uncompletedFileList, tmpDicomDir, tmpPredictionDir, bankDicomDir, bankPredictionDir):
    currentIndex = uncompletedFileList.index(specificEntry)
    dicomFoldersWeWant = {uncompletedFileList[currentIndex % len(uncompletedFileList)]}
                   #uncompletedFileList[currentIndex +1 % len(uncompletedFileList)]}
                   #uncompletedFileList[currentIndex +2 % len(uncompletedFileList)]}
    dicomFoldersWeHave = set(os.listdir(tmpDicomDir))
    if dicomFoldersWeWant == dicomFoldersWeHave:
        print('File system up to date')
        return True
    else:
        if not os.path.exists(bankDicomDir):
            return False
        else:
            print('Updating file system...')
            predictionFoldersWeWant = {uncompletedFileList[currentIndex % len(uncompletedFileList)][0:2] + '_processed'}
                                        #uncompletedFileList[currentIndex + 1 % len(uncompletedFileList)][0:2] + '_processed'}
                                        #uncompletedFileList[currentIndex + 2 % len(uncompletedFileList)][0:2] + '_processed'}
            predictionFoldersWeHave = set(os.listdir(tmpPredictionDir))
            #for newlyMadePredictionFolder in list(predictionFoldersWeHave - predictionFoldersWeWant):
            #    print('Moving folder ' + newlyMadePredictionFolder + ' to the permanent prediction bank storage')
            #    shutil.move(tmpPredictionDir + newlyMadePredictionFolder, bankPredictionDir + newlyMadePredictionFolder)
            for previouslyUsedDicomFolder in list(dicomFoldersWeHave - dicomFoldersWeWant):
                print('Deleting used dicom folder ' + previouslyUsedDicomFolder + ' from temporary storage')
                shutil.rmtree(tmpDicomDir + previouslyUsedDicomFolder)
            for dicomFolderWeWillNeed in list(dicomFoldersWeWant - dicomFoldersWeHave):
                print('Copying folder ' + dicomFolderWeWillNeed + ' to temporary storage')
                shutil.copytree(bankDicomDir + dicomFolderWeWillNeed, tmpDicomDir + dicomFolderWeWillNeed)
            return True

def main():

    indexStartLocations = {}
    boxSize = 256


    runTimeNum = '1'

    if get_mac() == 57277338463062:
        tmpStorageDir = 'C:/Users/Luke/Documents/sharedFolder/4YP/4YP_Pythoon/temporaryStorage' + runTimeNum +'/'
        tmpFolder = 'C:/Users/Luke/Documents/sharedFolder/4YP/4YP_Pythoon/tmp' + runTimeNum + '/'
        model_file = 'C:/Users/Luke/Documents/sharedFolder/4YP/Models/21stFeb/weights.43-0.01.h5'
        tmpDicomDir = 'C:/Users/Luke/Documents/sharedFolder/4YP/4YP_Pythoon/temporaryStorage' + runTimeNum + '/dicomFolders/'
        tmpPredictionDir = 'C:/Users/Luke/Documents/sharedFolder/4YP/4YP_Pythoon/temporaryStorage' + runTimeNum + '/predictionFolders/'
        bankDicomDir = 'D:/allCases/'
        bankPredictionDir = 'D:/processedCases/'
        [os.mkdir(myFolder) for myFolder in [tmpFolder, tmpStorageDir, tmpDicomDir, tmpPredictionDir] if not os.path.exists(myFolder)]
    else:
        tmpFolder = '//home/lukemarkham1383/segmentEnvironment/4YP_Python/tmp' + runTimeNum + '/'
        model_file = '//home/lukemarkham1383/segmentEnvironment/weights.43-0.01.h5'
        bankDicomDir = '//home/lukemarkham1383/segmentEnvironment/multipleScansGoodMachinesAortaOnlyContrasted/'
        bankPredictionDir = '//home/lukemarkham1383/segmentEnvironment/segmentedScans/'
    [os.mkdir(myFolder) for myFolder in [tmpFolder] if not os.path.exists(myFolder)]


    # Loads the model
    model = load_model(model_file)
    patientsWeHaveSegmented = [x[0:2] for x in os.listdir(bankPredictionDir)]
    patientsToSegmentList = sorted([x for x in os.listdir(bankDicomDir) if x[0:2] not in patientsWeHaveSegmented])
    patientsWeWant = patientsToSegmentList
    patientsWeWant = [x[0:2] for x in patientsWeWant]
    patientsToSegmentList = sorted([x for x in os.listdir(bankDicomDir) if x[0:2] in patientsWeWant])
    print(patientsToSegmentList)


    for patientNum, specificEntry in enumerate(patientsToSegmentList):
        print('Starting loop at ' + str(time.time()))
        print('Working on patient ' + str(patientNum+1) +'/'+str(len(patientsToSegmentList)))
        patientID = specificEntry[0:2]
        indexStartLocation = indexStartLocations[patientID] if patientID in indexStartLocations.keys() else 0

        innerPC, outerPC, patientID, predictionFolder = doPatientSegmentationWithoutStorage(specificEntry, patientsToSegmentList, indexStartLocation, model, boxSize, tmpFolder, bankDicomDir, bankPredictionDir)
        gc.collect()
        innerPC, outerPC, patientID, predictionFolder, innerDist = makeOrPopulateFolders(innerPC, outerPC, patientID, predictionFolder)
        gc.collect()
        cleanInnerPointCloud(innerPC, patientID, predictionFolder, innerDist)
        gc.collect()
        cleanOuterPointCloud(outerPC, patientID, predictionFolder, innerDist)
        gc.collect()
        subprocess.call('//home/lukemarkham1383/gdrive-linux-x64 upload //home/lukemarkham1383/segmentEnvironment/segmentLog1.txt', shell=True)

if __name__ == '__main__':
    main()
