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

dicomFolder = '/media/sf_sharedFolder/Images/39894NS/PreAugmentation/dicoms/'
model_file = '/media/sf_sharedFolder/Models/aaa/20thNov/weights.07-0.01.h5'

#Loads the model
model = load_model(model_file)

#Initialises arrays for the input from the dicom, the cropped dicom and the model input array
inputImage = np.ndarray([512, 512])
croppedImage = np.ndarray([256, 256])
modelInputArray = np.ndarray((1, 256, 256, 1), dtype='float32')

#Only checks files with an 80 in the name, meaning it checks IMG0080, IMG0180, IMG00280 etc, this is because checking every one is unnecessary
fileList = sorted(os.listdir(dicomFolder))
sampleFileList = filter(lambda k: '80' in k, fileList)

i = 0
xTotal = 0
yTotal = 0
for filename in sampleFileList:
    i = i + 1
    image = dicom.read_file(dicomFolder + filename)
    inputImage[:, :] = image.pixel_array

    #Gets image centre of mass, note y coordinate comes first and then x coordinate
    CoM = ndimage.measurements.center_of_mass(inputImage)
    xTotal = xTotal + CoM[1]
    yTotal = yTotal + CoM[0]

xAvg = math.floor(xTotal/i)
yAvg = math.floor(yTotal/i)

#Sets the limits for a 256x256 bounding box
xMin = int(xAvg - 128 if xAvg - 128 > 0 else 0)
xMax = int(xMin + 256)
yMin = int(yAvg - 128 if yAvg - 128 > 0 else 0)
yMax = int(yMin + 256)

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

    plt.subplot(121)
    plt.imshow(croppedImage, cmap='gray')
    plt.subplot(122)
    plt.imshow(predictedSegment, cmap='gray')
    plt.show()