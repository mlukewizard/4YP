from __future__ import print_function
import dicom
import os, shutil
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
from PIL import Image, ImageEnhance

tmpFolder = '/media/sf_sharedFolder/4YP_Python/tmp/'
dicomFolder = '/media/sf_sharedFolder/Images/NS/preAugmentation/dicoms/'
model_file = '/media/sf_sharedFolder/Models/28thNov/weights.05-0.04.h5'
outputPredictions = '/media/sf_sharedFolder/predictions/'

try:
    shutil.rmtree(tmpFolder)
except:
    pass
os.mkdir(tmpFolder)

#Loads the model
model = load_model(model_file)

#Initialises arrays for the input from the dicom, the cropped dicom and the model input array
inputImage = np.ndarray([512, 512], dtype='float32')
croppedImage = np.ndarray([256, 256], dtype='float32')

#Gets the locations of the CoM so you know where to chop the dicoms
[xMin, xMax, yMin, yMax] = getFolderCoM(dicomFolder)

#This bit makes the folder into a numpy array
patientID = 'NS'
imageType = 'Original'
counter = 0

fileList = sorted(os.listdir(dicomFolder))
imgTotal = len(fileList)

npImageArray = np.ndarray((imgTotal, 5, 256, 256, 1), dtype='float32')

print('Turning files into numpy arrays')
for filename in fileList:
    print('Reading file ' + str(counter) + '/' + str(imgTotal))
    # Read the dicom into a png
    inputDicomImage = dicom.read_file(dicomFolder + filename)
    inputImage[:, :] = inputDicomImage.pixel_array
    misc.imsave('/media/sf_sharedFolder/4YP_Python/tmp/dicomTemp.png', inputImage)
    croppedImage = misc.imread('/media/sf_sharedFolder/4YP_Python/tmp/dicomTemp.png')[yMin:yMax, xMin:xMax]
    image = Image.fromarray((croppedImage))
    os.remove(tmpFolder + 'dicomTemp.png')

    if counter > 3 and counter < imgTotal - 4:
        #assign to this index
        npImageArray[counter, 2, :, :, 0] = image

        #assign to previous indexes
        npImageArray[counter-2, 3, :, :, 0] = image
        npImageArray[counter-4, 4, :, :, 0] = image

        #assign to future indexes
        npImageArray[counter+2, 1, :, :, 0] = image
        npImageArray[counter+4, 0, :, :, 0] = image

    elif counter > 1 and counter < 4: #gets index 2 and 3
        #assign to this index
        npImageArray[counter, 2, :, :, 0] = image
        npImageArray[counter, 1, :, :, 0] = image
        npImageArray[counter, 0, :, :, 0] = image #this is done for contingency

        #assign to previous indexes
        npImageArray[counter - 2, 3, :, :, 0] = image

        # assign to future indexes
        npImageArray[counter + 2, 1, :, :, 0] = image
        npImageArray[counter + 4, 0, :, :, 0] = image

    elif counter < 2: #gets indexes 0 and 1
        # assign to this index
        npImageArray[counter, 2, :, :, 0] = image
        npImageArray[counter, 1, :, :, 0] = image
        npImageArray[counter, 0, :, :, 0] = image  # this is necessary

        # assign to future indexes
        npImageArray[counter + 2, 1, :, :, 0] = image
        npImageArray[counter + 4, 0, :, :, 0] = image

    elif counter > imgTotal - 5 and counter < imgTotal - 2: #gets indexes imgtotal-3 and imgtotal-4
        # assign to this index
        npImageArray[counter, 2, :, :, 0] = image
        npImageArray[counter, 3, :, :, 0] = image
        npImageArray[counter, 4, :, :, 0] = image  # this is done for contingency

        # assign to previous indexes
        npImageArray[counter - 2, 3, :, :, 0] = image
        npImageArray[counter - 4, 4, :, :, 0] = image

        # assign to future indexes
        npImageArray[counter + 2, 1, :, :, 0] = image

    elif counter > imgTotal - 3: #gets the end and the one before it
        #assigns to this index
        npImageArray[counter, 2, :, :, 0] = image
        npImageArray[counter, 3, :, :, 0] = image
        npImageArray[counter, 4, :, :, 0] = image #this is needed

        #assigns to prevous indexes
        npImageArray[counter - 2, 3, :, :, 0] = image
        npImageArray[counter - 4, 4, :, :, 0] = image

    counter = counter + 1

#for i in range(imgTotal):
#    plt.imshow(npImageArray[i, 2, :, :, 0])
#    plt.show()

predictedImageArray = np.ndarray((imgTotal, 5, 256, 256, 2), dtype='float32')
modelInputArray = np.ndarray((1, 5, 256, 256, 1), dtype='float32')

print('Starting predictions')
for i in range(0, imgTotal, 20):
    # Predicts the location of the aneurysm
    print("Predicting slice " + str(i) + '/' + str(imgTotal))
    modelInputArray[:,:,:,:,:] = npImageArray[i,:,:,:,:]
    predictedImageArray[i,:,:,:,:] = model.predict(modelInputArray)

    #You should save down the predicted binaries here
    plt.subplot(131)
    plt.imshow(npImageArray[i,2,:,:,0], cmap='gray')
    plt.subplot(132)
    plt.imshow(predictedImageArray[i,2,:,:,0], cmap='gray')
    plt.subplot(133)
    plt.imshow(predictedImageArray[i,2,:,:,1], cmap='gray')
    plt.show()

