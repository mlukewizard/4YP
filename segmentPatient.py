from __future__ import print_function
from imageProcessingFuntions import getFolderCoM, lukesAugment, Construct3DDicomArray
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
import scipy.misc as misc
from PIL import Image, ImageEnhance

tmpFolder = '/media/sf_sharedFolder/4YP/4YP_Python/tmp/'
dicomFolder = '/media/sf_sharedFolder/4YP/Images/GC_Dicoms/'
model_file = '/media/sf_sharedFolder/4YP/Models/18thDec/weights.05-0.05.h5'
outputPredictions = '/media/sf_sharedFolder/4YP/predictions/'

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
patientID = 'GC'
imageType = 'Original'
counter = 0

fileList = sorted(os.listdir(dicomFolder))
imgTotal = len(fileList)

npImageArray = np.ndarray((imgTotal, 5, 256, 256, 1), dtype='float32')

print('Turning files into numpy arrays')
for filename in fileList:
    print('Reading file ' + str(counter) + '/' + str(imgTotal))
    counter = counter + 1
    # Read the dicom into a png
    inputDicomImage = dicom.read_file(dicomFolder + filename)
    inputImage[:, :] = inputDicomImage.pixel_array
    misc.imsave(tmpFolder + filename.split('.')[0] + '.png', inputImage)
    croppedImage = misc.imread(tmpFolder + filename.split('.')[0] + '.png')[yMin:yMax, xMin:xMax]
    os.remove(tmpFolder+ filename.split('.')[0] + '.png')
    croppedImage = Image.fromarray((croppedImage))
    croppedImage = lukesAugment(croppedImage)
    croppedImage.convert('RGB').save(tmpFolder + 'Original' + '%.3d' % counter + 'Patient' + patientID + '.png', 'PNG')

print('Saved all')

npImageArray = Construct3DDicomArray(tmpFolder, '/media/sf_sharedFolder/4YP/npArrays', patientID, True, 1, True, False)

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

