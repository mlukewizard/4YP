from __future__ import print_function
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
from imageProcessingFuntions import getFolderCoM
import scipy.misc as misc

dicomFolder = '/media/sf_sharedFolder/Images/39894NS/PreAugmentation/dicoms/'
model_file = '/media/sf_sharedFolder/Models/20thNov/weights.07-0.01.h5'
outputPredictions = '/media/sf_sharedFolder/predictions/'

#Loads the model
model = load_model(model_file)

#Initialises arrays for the input from the dicom, the cropped dicom and the model input array
inputImage = np.ndarray([512, 512])
croppedImage = np.ndarray([256, 256])
modelInputArray = np.ndarray((1, 256, 256, 1), dtype='float32')

#Gets the locations of the CoM so you know where to chop the dicoms
[xMin, xMax, yMin, yMax] = getFolderCoM(dicomFolder)

fileList = sorted(os.listdir(dicomFolder))
fileList = filter(lambda k: '60' in k, fileList) #delete this!
for filename in fileList:
    image = dicom.read_file(dicomFolder + filename)

    #Takes the data out of the weird dicom format and puts it in a numpy array
    inputImage[:,:] = image.pixel_array

    #Crops the image
    croppedImage[:,:] = inputImage[yMin:yMax, xMin:xMax]

    #Assigns the cropped image to the centre of the numpy array
    modelInputArray[0,:,:,0] = croppedImage

    #Predicts the location of the aneurysm
    predictedSegment = model.predict(modelInputArray)
    predictedSegment = predictedSegment[0, :, :, 0]
    misc.imsave(outputPredictions + filename + 'outfile.png', predictedSegment)

    plt.subplot(121)
    plt.imshow(croppedImage, cmap='gray')
    plt.subplot(122)
    plt.imshow(predictedSegment, cmap='gray')
    plt.show()