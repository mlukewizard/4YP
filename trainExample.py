from __future__ import print_function
from __future__ import division
from keras.callbacks import ModelCheckpoint
from keras.models import Model, load_model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, ConvLSTM2D, LSTM, TimeDistributed, Bidirectional, Dropout, BatchNormalization, Activation
from keras.optimizers import Adam
from keras import losses
import numpy as np
import os
import sys
import h5py
from keras import optimizers
from random import *
from myModels import *
from keras import backend as K

#Defining loss only for the middle slice (if needed)
def my_loss(y_true, y_pred):
    return K.mean(K.binary_crossentropy(y_true[:, 2, :, :, :], y_pred[:, 2, :, :, :]))
losses.my_loss = my_loss

#Program inputs
twoDVersion = False
patientList = ['PS', 'PB', 'RR', 'DC']
trainingArrayDepth = 300 
augmentationsInTrainingArray = len(patientList)
boxSize = 256

#Initilising training arrays
if not twoDVersion:
    img_measure = np.ndarray((trainingArrayDepth, 5, boxSize, boxSize, 1), dtype='float32')
    bm_measure = np.ndarray((trainingArrayDepth, 5, boxSize, boxSize, 2), dtype='float32')
else:
    img_measure = np.ndarray((trainingArrayDepth, boxSize, boxSize, 1), dtype='float32')
    bm_measure = np.ndarray((trainingArrayDepth, boxSize, boxSize, 2), dtype='float32')

#Defining file paths
validationArrayPath = '/home/lukemarkham1383/trainEnvironment/npArrays/validation/'
trainingArrayPath = '/home/lukemarkham1383/trainEnvironment/npArrays/training/'
model_folder = '/home/lukemarkham1383/trainEnvironment/models/'

if not twoDVersion:
    img_test_file = '3DNonAugmentPatientNS_dicom.npy'
    bm_test_file = '3DNonAugmentPatientNS_binary.npy'
else:
    img_test_file = '2DNonAugmentPatientNS_dicom.npy'
    bm_test_file = '2DNonAugmentPatientNS_binary.npy'

#Loading numpy arrays for validation
img_test = np.load(os.path.join(validationArrayPath, img_test_file))
bm_test = np.load(os.path.join(validationArrayPath, bm_test_file)) / 255

# Defining the split of slices from each of the randomly selected arrays
arraySplits = np.linspace(0, trainingArrayDepth, len(patientList)+1, dtype = 'int')

fileList = sorted(os.listdir(trainingArrayPath))
dicomFileList = filter(lambda k: 'dicom' in k, fileList)
binaryFileList = filter(lambda k: 'bina' in k, fileList)
for k in range(30):
    print('Constructing Arrays')  

    for i in range(len(arraySplits)-1):
        # This while loop ensures you get a patient from each patient for each epoch
        patientFile = 'RandomString'
        while patientFile.find(patientList[i]) == -1:
            shuffle(dicomFileList)
            patientFile = dicomFileList[0]
        print('Using data from ' + patientFile)

        # Loads arrays of images
        dicomFile = np.load(trainingArrayPath + patientFile)
        binaryFile = np.load(trainingArrayPath + patientFile.split("dic")[0] + 'binary.npy') / 255

        # Randomly writes to the training arrays from the contents of the arrays of images
        for j in range(arraySplits[i+1]-arraySplits[i]):
            index = int(np.round(uniform(0, len(dicomFile)-1)))
            if not twoDVersion:
                img_measure[arraySplits[i]+j, :, :, :, :] = dicomFile[index]
                bm_measure[arraySplits[i]+j, :, :, :, :] = binaryFile[index]
            else:
                img_measure[arraySplits[i] + j, :, :, :] = dicomFile[index]
                bm_measure[arraySplits[i] + j, :, :, :] = binaryFile[index]

    # Defines the test split so that your validtion array doesnt feature in the training set
    testSplit = img_test.shape[0]/(img_test.shape[0]+img_measure.shape[0])
    print('Validation split is ' + str(testSplit))

    # Concatenates the two arrays
    img_train = np.concatenate((img_measure, img_test))
    bm_train = np.concatenate((bm_measure, bm_test))

    print('Building Model')
    model_list = os.listdir(model_folder)  # Checking if there is an existing model
    if model_list.__len__() == 0:  # Creating a new model if empty

        # Get the model you want to use from the models bank
        if not twoDVersion:
            model = my3DModel(boxSize)
        else:
            model = my2DModel(boxSize)

        # If there isnt a previous number then this epoch must be epoch number 0
        epoch_number = 0

    else:
        #Scrolls through the model list and find the model with the highest epoch number
        currentMax = 0
        for fn in model_list:
            epoch_number = int(fn.split('weights.')[1].split('-')[0])
            if epoch_number > currentMax:
                currentMax = epoch_number
                model_file = fn
        epoch_number = int(model_file.split('weights.')[1].split('-')[0])

        #Loads that model file
        f_model = h5py.File(os.path.join(model_folder, model_file), 'r+')
        if 'optimizer_weights' in f_model:
            del f_model['optimizer_weights']
        f_model.close()

        #Loads the model from that file
        model = load_model(os.path.join(model_folder, model_file))
        print('Using model number ' + str(epoch_number))

    # Defines the compile settings
    #if not twoDVersion:
	#model.compile(optimizer=Adam(lr=1e-3), loss=my_loss)
    #else:
    model.compile(optimizer=Adam(lr=1e-3), loss=losses.binary_crossentropy)

    #Defines the checkpoint file
    model_check_file = os.path.join(model_folder, 'weights.{epoch:02d}-{loss:.2f}.h5')
    model_checkpoint = ModelCheckpoint(model_check_file, monitor='val_loss', save_best_only=False)

    #Actually do the training for this epoch
    print('Starting train')
    if not twoDVersion:
        myBatchSize = 2
    else:
        myBatchSize = 4
    model.fit(img_train, bm_train, batch_size=myBatchSize, initial_epoch=epoch_number, epochs=epoch_number + 1,
                            verbose=1, shuffle=True, validation_split=testSplit,
                            callbacks=[model_checkpoint])
