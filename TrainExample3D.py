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
from keras import backend as K
from random import *

patientList = ['PS', 'PB', 'RR', 'DC']
trainingArrayDepth = 300 
augmentationsInTrainingArray = len(patientList)
boxSize = 150

img_measure = np.ndarray((trainingArrayDepth, 5, boxSize, boxSize, 1), dtype='float32')
bm_measure = np.ndarray((trainingArrayDepth, 5, boxSize, boxSize, 2), dtype='float32')

def my_loss(y_true, y_pred):
    return K.mean(K.binary_crossentropy(y_true[:,2,:,:,:], y_pred[:,2,:,:,:]))

losses.my_loss = my_loss

validationArrayPath = '/home/lukemarkham1383/trainEnvironment/npArrays/validation/'
trainingArrayPath = '/home/lukemarkham1383/trainEnvironment/npArrays/training/'
model_folder = '/home/lukemarkham1383/trainEnvironment/models/'

img_test_file = '3DNonAugmentPatientNS_dicom.npy'
bm_test_file = '3DNonAugmentPatientNS_binary.npy'
img_test = np.load(os.path.join(validationArrayPath, img_test_file))
bm_test = np.load(os.path.join(validationArrayPath, bm_test_file)) / 255

arraySplits = np.linspace(0, trainingArrayDepth, len(patientList)+1, dtype = 'int')

fileList = sorted(os.listdir(trainingArrayPath))
dicomFileList = filter(lambda k: 'dicom' in k, fileList)
binaryFileList = filter(lambda k: 'bina' in k, fileList)
for k in range(10):
    print('Constructing Arrays')  

    for i in range(len(arraySplits)-1):
	patientFile = 'RandomString'
	while patientFile.find(patientList[i]) == -1:
		shuffle(dicomFileList)
		patientFile = dicomFileList[0]
	print('Using data from ' + patientFile)
	dicomFile = np.load(trainingArrayPath + patientFile)
	binaryFile = np.load(trainingArrayPath + patientFile.split("dic")[0] + 'binary.npy') / 255
	#print('Printing for ' + str(arraySplits[i+1]-arraySplits[i]))
        for j in range(arraySplits[i+1]-arraySplits[i]):
            index = int(np.round(uniform(0, len(dicomFile)-1)))
	    #print('Printing at index ' + str(arraySplits[i]+j))
	    #print('Printing from index ' + str(index))
            img_measure[arraySplits[i]+j, :, :, :, :] = dicomFile[index]
            bm_measure[arraySplits[i]+j, :, :, :, :] = binaryFile[index]

    #np.save(model_folder + 'TestBinary' + '.npy', bm_train)
    #np.save(model_folder + 'TestDicom' + '.npy', img_train)
    #sys.exit()
    
    testSplit = img_test.shape[0]/(img_test.shape[0]+img_measure.shape[0])
    print('Validation split is ' + str(testSplit))

    img_train = np.concatenate((img_measure, img_test))[:,:,0:144, 0:144, :]
    bm_train = np.concatenate((bm_measure, bm_test))[:,:,0:144, 0:144, :]
	
    #np.save(model_folder + 'TestBinary' + '.npy', bm_train)
    #np.save(model_folder + 'TestDicom' + '.npy', img_train)
    #sys.exit()

    print('Building Model')	

    model_list = os.listdir(model_folder)  # Checking if there is an existing model
    if model_list.__len__() == 0:  # Creating a new model if empty

        inputs = Input((5, 160, 160, 1))
	
	conv1 = TimeDistributed(BatchNormalization())(inputs)
        conv1 = TimeDistributed(Conv2D(32, (3, 3), padding='same'))(conv1)
	conv1 = TimeDistributed(BatchNormalization())(conv1)
	conv1 = TimeDistributed(Activation('relu'))(conv1)
	conv1 = TimeDistributed(Dropout(0.2))(conv1)
        pool1 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv1)
	pool1 = TimeDistributed(Dropout(0.2))(pool1)	

        conv2 = TimeDistributed(Conv2D(64, (3, 3), padding='same'))(pool1)
        conv2 = TimeDistributed(BatchNormalization())(conv2)
	conv2 = TimeDistributed(Activation('relu'))(conv2)
        conv2 = TimeDistributed(Dropout(0.2))(conv2)
        pool2 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv2)
	pool2 = TimeDistributed(Dropout(0.2))(pool2)

        conv3 = TimeDistributed(Conv2D(128, (3, 3), padding='same'))(pool2)
        conv3 = TimeDistributed(BatchNormalization())(conv3)
	conv3 = TimeDistributed(Activation('relu'))(conv3)
        conv3 = TimeDistributed(Dropout(0.2))(conv3)
        pool3 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv3)
	pool3 = TimeDistributed(Dropout(0.2))(pool3)

        conv4 = TimeDistributed(Conv2D(256, (3, 3), padding='same'))(pool3)
        conv4 = TimeDistributed(BatchNormalization())(conv4)
        conv4 = TimeDistributed(Activation('relu'))(conv4)
        conv4 = TimeDistributed(Dropout(0.2))(conv4)
        pool4 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv4)
        pool4 = TimeDistributed(Dropout(0.2))(pool4)

        myLSTM = Bidirectional(ConvLSTM2D(512, (3, 3), activation='relu', padding='same', return_sequences=True))(pool4)

        up6 = concatenate([TimeDistributed(UpSampling2D(size=(2, 2)))(myLSTM), conv4], axis=4)
        conv6 = TimeDistributed(Conv2D(256, (3, 3), padding='same'))(up6)
	conv6 = TimeDistributed(BatchNormalization())(conv6)
	conv6 = TimeDistributed(Activation('relu'))(conv6)
        conv6 = TimeDistributed(Dropout(0.2))(conv6)

        up7 = concatenate([TimeDistributed(UpSampling2D(size=(2, 2)))(conv6), conv3], axis=4)
        conv7 = TimeDistributed(Conv2D(128, (3, 3), padding='same'))(up7)
        conv7 = TimeDistributed(BatchNormalization())(conv7)
        conv7 = TimeDistributed(Activation('relu'))(conv7)
        conv7 = TimeDistributed(Dropout(0.2))(conv7)

	up8 = concatenate([TimeDistributed(UpSampling2D(size=(2, 2)))(conv7), conv2], axis=4)
        conv8 = TimeDistributed(Conv2D(64, (3, 3), padding='same'))(up8)
        conv8 = TimeDistributed(BatchNormalization())(conv8)
        conv8 = TimeDistributed(Activation('relu'))(conv8)
        conv8 = TimeDistributed(Dropout(0.2))(conv8)

	up9 = concatenate([TimeDistributed(UpSampling2D(size=(2, 2)))(conv8), conv1], axis=4)
        conv9 = TimeDistributed(Conv2D(32, (3, 3), padding='same'))(up9)
        conv9 = TimeDistributed(BatchNormalization())(conv9)
        conv9 = TimeDistributed(Activation('relu'))(conv9)
        conv9 = TimeDistributed(Dropout(0.2))(conv9)

        conv10 = TimeDistributed(Conv2D(2, (1, 1)))(conv9)
	conv10 = TimeDistributed(BatchNormalization())(conv10)
	conv10 = TimeDistributed(Activation('sigmoid'))(conv10)

        model = Model(inputs=[inputs], outputs=[conv10])

        epoch_number = 0

    else:
        currentMax = 0
        for fn in model_list:
            epoch_number = int(fn.split('weights.')[1].split('-')[0])
            if epoch_number > currentMax:
                currentMax = epoch_number
                model_file = fn
        epoch_number = int(model_file.split('weights.')[1].split('-')[0])
        f_model = h5py.File(os.path.join(model_folder, model_file), 'r+')
        if 'optimizer_weights' in f_model:
            del f_model['optimizer_weights']
        f_model.close()
        model = load_model(os.path.join(model_folder, model_file))
        print('Using model number ' + str(epoch_number))

    #model.summary()
    #model.compile(optimizer=Adam(lr=1e-3), loss=losses.binary_crossentropy)
    model.compile(optimizer=Adam(lr=1e-3), loss=my_loss)
    model_check_file = os.path.join(model_folder, 'weights.{epoch:02d}-{loss:.2f}.h5')
    model_checkpoint = ModelCheckpoint(model_check_file, monitor='val_loss', save_best_only=False)
    print('Starting train')
    model.fit(img_train, bm_train, batch_size=2, initial_epoch=epoch_number, epochs=epoch_number + 1,
                            verbose=1, shuffle=True, validation_split=testSplit,
                            callbacks=[model_checkpoint])
