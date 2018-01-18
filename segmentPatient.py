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
from keras import backend as K
def my_loss(y_true, y_pred):
    return K.mean(K.binary_crossentropy(y_true[:, 2, :, :, :], y_pred[:, 2, :, :, :]))
losses.my_loss = my_loss


patientList = ['MH']
indexStartLocations = [300] #160
centralCoordinates = [[200, 280]]#[[310, 286]] # In form [yPosition, xPosition] (remember axis is from top left)
boxSize = 144

tmpFolder = 'C:\\Users\\Luke\\Documents\\sharedFolder\\4YP\\4YP_Python\\tmp\\'
if not os.path.exists(tmpFolder):
            os.mkdir(tmpFolder)
model_file = 'C:\\Users\\Luke\\Documents\\sharedFolder\\4YP\\Models\\17thJan\\weights.56-0.03.h5'

# Loads the model
model = load_model(model_file)

for patientID, indexStartLocation, centralCoordinate in zip(patientList, indexStartLocations, centralCoordinates):
    dicomFolder = 'C:\\Users\\Luke\\Documents\\sharedFolder\\4YP\\Images\\' + patientID + '_dicoms\\'
    outputPredictions = 'C:\\Users\\Luke\\Documents\\sharedFolder\\4YP\\predictions\\' + patientID + 'MH\\'

    dicomList = sorted(os.listdir(dicomFolder))
    print('You should check the following are in alphabetical/numerical order')
    print(dicomList[0])
    print(dicomList[1])
    print(dicomList[2])
    for k in range(indexStartLocation, len(dicomList)):
        # Predicts the location of the aneurysm
        print("Predicting slice " + str(k) + '/' + str(len(dicomList)))

        #dicomImage = dicom.read_file(dicomFolder + dicomList[indexStartLocation]).pixel_array
        #misc.imsave(tmpFolder + 'dicomTemp.png', dicomImage)
        #plt.imshow(misc.imread(tmpFolder + 'dicomTemp.png'))
        #plt.show()
        #os.remove(tmpFolder + 'dicomTemp.png')

        modelInputArray = ConstructArraySlice(dicomList, dicomFolder, k, boxSize, centralLocation=centralCoordinate, twoDVersion = True)
        output = model.predict(np.expand_dims(modelInputArray, axis=0))*255
        #newLocation = ndimage.measurements.center_of_mass(output[0, 2, :, :, 0])
        newLocation = ndimage.measurements.center_of_mass(output[0, :, :, 0])
        centralCoordinate = [int(centralCoordinate[0] - boxSize/2 + round(newLocation[0])), int(centralCoordinate[1] - boxSize/2 + round(newLocation[1]))]
        saveSlice(modelInputArray, output, saveFig = True, saveFolder = 'C:\\Users\\Luke\\Documents\\sharedFolder\\4YP\\savedFigures\\')
