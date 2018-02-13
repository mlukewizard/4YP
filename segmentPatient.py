from __future__ import print_function
from myFunctions import *
import dicom
import os
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
mac = get_mac()
from keras import backend as K
def my_loss(y_true, y_pred):
    return K.mean(K.binary_crossentropy(y_true[:, 2, :, :, :], y_pred[:, 2, :, :, :]))
losses.my_loss = my_loss

twoDVersion = False
patientList = ['MH', 'SC']
indexStartLocations = [0, 0]
centralCoordinates = [[240, 256], [240, 256]] # In form [yPosition, xPosition] (remember axis is from top left)
boxSize = 256

if mac == 57277338463062:
    tmpFolder = 'C:\\Users\\Luke\\Documents\\sharedFolder\\4YP\\4YP_Python\\tmp\\'
    if not os.path.exists(tmpFolder):
                os.mkdir(tmpFolder)
    model_file = 'C:\\Users\\Luke\\Documents\\sharedFolder\\4YP\Models\\12thFeb\\weights.15-0.01.h5'
    dicomSetFolder = 'C:\\Users\\Luke\\Documents\\sharedFolder\\4YP\\algoSegmentations\\'

# Loads the model
model = load_model(model_file)

for patientID, indexStartLocation, centralCoordinate in zip(patientList, indexStartLocations, centralCoordinates):
    dicomFolder = dicomSetFolder + patientID + '_dicoms\\'
    predictionFolder = dicomSetFolder + patientID + '_predictions\\'
    innerPredictionFolder = predictionFolder + 'innerPredictions\\'
    outerPredictionFolder = predictionFolder + 'outerPredictions\\'
    dicomToPngFolder = predictionFolder + 'pngs\\'

    dicomList = sorted(os.listdir(dicomFolder))
    print('You should check the following are in alphabetical/numerical order')
    print(dicomList[0])
    print(dicomList[1])
    print(dicomList[2])

    if not all(os.path.exists(myFolder) for myFolder in [dicomToPngFolder + patientID + '_dicomArray' + '.npy', innerPredictionFolder + patientID + '_innerBinaryArray' + '.npy',
                                                     outerPredictionFolder + patientID + '_outerBinaryArray' + '.npy', dicomToPngFolder]):
        print('No previously made numpy arrays found')
        [os.mkdir(myFolder) for myFolder in [predictionFolder, innerPredictionFolder, outerPredictionFolder, dicomToPngFolder] if not os.path.exists(myFolder)]

        #Initialises the pointCloud
        innerPC = np.zeros([len(dicomList), 512, 512])
        outerPC = np.zeros([len(dicomList), 512, 512])
        dicomPC = np.zeros([len(dicomList), 512, 512])

        for loopCount, k in enumerate(range(len(dicomList))):
            #Predicts the location of the aneurysm
            print("Predicting slice " + str(k) + '/' + str(len(dicomList)))

            #Constructing a suitable array to feeed to the algorithm so it can segment the slice in question
            if not twoDVersion:
                modelInputArray = np.expand_dims(ConstructArraySlice(dicomList, dicomFolder, k, boxSize, centralLocation=centralCoordinate), axis=0)
            else:
                modelInputArray = np.expand_dims(ConstructArraySlice(dicomList, dicomFolder, k, boxSize, centralLocation=centralCoordinate, twoDVersion=True), axis=0)

            #Uses the algorithm to predict the location of the aneurysm
            output = model.predict(modelInputArray)*255

            #Gets the location of the 256x256 box in the 512x512 image
            upperRow = centralCoordinate[0] - round(boxSize / 2)
            lowerRow = upperRow + boxSize
            leftColumn = centralCoordinate[1] - round(boxSize / 2)
            rightColumn = leftColumn + boxSize

            #Gets a copy of the original image you're trying to segment
            dicomImage = dicom.read_file(dicomFolder + dicomList[k]).pixel_array
            misc.toimage(255 * (dicomImage / 4095), cmin=0.0, cmax=255).save(tmpFolder + 'dicomTemp.png')
            dicomImage = misc.imread(tmpFolder + 'dicomTemp.png', flatten=True)
            os.remove(tmpFolder + 'dicomTemp.png')
            dicomImage = lukesAugment(dicomImage)

            if not twoDVersion:
                #misc.toimage(dicomImage, cmin=0.0, cmax=255).save((dicomToPngFolder + 'dicomToPng' + dicomList[k]).split('.dcm')[0] + '.png')
                resizedOuterImage = np.zeros([512, 512])
                resizedOuterImage[upperRow:lowerRow, leftColumn:rightColumn] = output[0, 2, :, :, 1]
                #misc.toimage(resizedOuterImage, cmin=0.0, cmax=255).save((outerPredictionFolder + 'outerPredicted_' + dicomList[k]).split('.dcm')[0] + '.png')
                resizedInnerImage = np.zeros([512, 512])
                resizedInnerImage[upperRow:lowerRow, leftColumn:rightColumn] = output[0, 2, :, :, 0]
                #misc.toimage(resizedInnerImage, cmin=0.0, cmax=255).save((innerPredictionFolder + 'innerPredicted_' + dicomList[k]).split('.dcm')[0] + '.png')

                outerPC[k, :, :] = resizedOuterImage
                innerPC[k, :, :] = resizedInnerImage
                dicomPC[k, :, :] = dicomImage
            else:
                print('worry about this another day')

            # Updates the location of the centre of mass for the next iteration
            if not twoDVersion:
                newLocation = ndimage.measurements.center_of_mass(resizedInnerImage + resizedOuterImage)
            else:
                newLocation = ndimage.measurements.center_of_mass(resizedInnerImage + resizedOuterImage)
            centralCoordinate = [int(centralCoordinate[0] + (newLocation[0]-centralCoordinate[0])/np.power(loopCount+1, 0.2)), int(centralCoordinate[1] + (newLocation[1]-centralCoordinate[1])/np.power(loopCount+1, 0.2))]
            #centralCoordinate = [int(centralCoordinate[0] - boxSize / 2 + round(newLocation[0])), int(centralCoordinate[1] - boxSize / 2 + round(newLocation[1]))]

        np.save(outerPredictionFolder + patientID + '_outerBinaryArray' + '.npy', outerPC)
        np.save(innerPredictionFolder + patientID + '_innerBinaryArray' + '.npy', innerPC)
        np.save(dicomToPngFolder + patientID + '_dicomArray' + '.npy', dicomPC)
    else:
        print('Using previously constructed numpy arrays')
        dicomPC = np.load(dicomToPngFolder + patientID + '_dicomArray' + '.npy')
        innerPC = np.load(innerPredictionFolder + patientID + '_innerBinaryArray' + '.npy')
        outerPC = np.load(outerPredictionFolder + patientID + '_outerBinaryArray' + '.npy')

    for i in range(outerPC.shape[0]):
        misc.toimage(outerPC[i, :, :], cmin=0.0, cmax=255).save((outerPredictionFolder + 'outerPredicted_' + dicomList[i]).split('.dcm')[0] + '.png')
    for i in range(innerPC.shape[0]):
        misc.toimage(innerPC[i, :, :], cmin=0.0, cmax=255).save((innerPredictionFolder + 'innerPredicted_' + dicomList[i]).split('.dcm')[0] + '.png')
    for i in range(outerPC.shape[0]):
        misc.toimage(dicomPC[i, :, :], cmin=0.0, cmax=255).save((dicomToPngFolder + 'dicomToPng_' + dicomList[i]).split('.dcm')[0] + '.png')
