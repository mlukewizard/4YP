from __future__ import print_function
from myFunctions import getFolderCoM, lukesAugment, Construct3DDicomArray
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

patientID = 'MH'
tmpFolder = '/media/sf_sharedFolder/4YP/4YP_Python/tmp/'
dicomFolder = '/media/sf_sharedFolder/4YP/Images/' + patientID + '_dicoms/'
model_file = '/media/sf_sharedFolder/4YP/Models/2ndJan/weights.04-0.11.h5'
outputPredictions = '/media/sf_sharedFolder/4YP/predictions/'
modelTestArrayDir = '/media/sf_sharedFolder/4YP/npArrays/modelTestArray/'

try:
    shutil.rmtree(tmpFolder)
except:
    pass
os.mkdir(tmpFolder)

#Loads the model
model = load_model(model_file)

gotArray = False
for filename in sorted(os.listdir(modelTestArrayDir)):
    if filename.find(patientID) != -1:
        gotArray = True

if gotArray == True:
    print('Using already defined test array')
    fileList = sorted(os.listdir(modelTestArrayDir))
    modelTestArrayFile = fileList[0]
    modelTestArray = np.load(modelTestArrayDir + modelTestArrayFile)
    imgTotal = modelTestArray.shape[0]

else:

    #Initialises arrays for the input from the dicom, the cropped dicom and the model input array
    inputImage = np.ndarray([512, 512], dtype='float32')
    croppedImage = np.ndarray([256, 256], dtype='float32')

    #Gets the locations of the CoM so you know where to chop the dicoms
    [xMin, xMax, yMin, yMax] = getFolderCoM(dicomFolder)

    #This bit makes the folder into a numpy array
    imageType = 'Original'
    counter = 0

    fileList = sorted(os.listdir(dicomFolder))
    imgTotal = len(fileList)

    npImageArray = np.ndarray((imgTotal, 5, 256, 256, 1), dtype='float32')

    print('Turning files into dicom numpy arrays')
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
    modelTestArray = Construct3DDicomArray(tmpFolder, '/media/sf_sharedFolder/4YP/npArrays', patientID, True, 1, True, False)
    np.save(modelTestArrayDir + patientID + 'TestArray.npy', modelTestArray)

predictedImageArray = np.ndarray((imgTotal, 5, 256, 256, 2), dtype='float32')
modelInputArray = np.ndarray((1, 5, 256, 256, 1), dtype='float32')

print('Starting predictions')
for k in range(300, imgTotal, 20):
    # Predicts the location of the aneurysm
    print("Predicting slice " + str(k) + '/' + str(imgTotal))
    modelInputArray[:,:,:,:,:] = modelTestArray[k,:,:,:,:]
    predictedImageArray[k,:,:,:,:] = model.predict(modelInputArray)*255

    #You should save down the predicted binaries here
    plt.subplot(3, 5, 1)
    plt.imshow(modelTestArray[k, 0, :, :, 0], cmap='gray')
    plt.subplot(3, 5, 2)
    plt.imshow(modelTestArray[k, 1, :, :, 0], cmap='gray')
    plt.subplot(3, 5, 3)
    plt.imshow(modelTestArray[k, 2, :, :, 0], cmap='gray')
    plt.subplot(3, 5, 4)
    plt.imshow(modelTestArray[k, 3, :, :, 0], cmap='gray')
    plt.subplot(3, 5, 5)
    plt.imshow(modelTestArray[k, 4, :, :, 0], cmap='gray')
    plt.subplot(3, 5, 6)
    plt.imshow(predictedImageArray[k, 0, :, :, 0], cmap='gray')
    plt.subplot(3, 5, 7)
    plt.imshow(predictedImageArray[k, 1, :, :, 0], cmap='gray')
    plt.subplot(3, 5, 8)
    plt.imshow(predictedImageArray[k, 2, :, :, 0], cmap='gray')
    plt.subplot(3, 5, 9)
    plt.imshow(predictedImageArray[k, 3, :, :, 0], cmap='gray')
    plt.subplot(3, 5, 10)
    plt.imshow(predictedImageArray[k, 4, :, :, 0], cmap='gray')
    plt.subplot(3, 5, 11)
    plt.imshow(predictedImageArray[k, 0, :, :, 1], cmap='gray')
    plt.subplot(3, 5, 12)
    plt.imshow(predictedImageArray[k, 1, :, :, 1], cmap='gray')
    plt.subplot(3, 5, 13)
    plt.imshow(predictedImageArray[k, 2, :, :, 1], cmap='gray')
    plt.subplot(3, 5, 14)
    plt.imshow(predictedImageArray[k, 3, :, :, 1], cmap='gray')
    plt.subplot(3, 5, 15)
    plt.imshow(predictedImageArray[k, 4, :, :, 1], cmap='gray')
    plt.show()
