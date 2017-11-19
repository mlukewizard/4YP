from __future__ import print_function
from keras.callbacks import ModelCheckpoint
from keras.models import Model, load_model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from keras.optimizers import Adam
from keras import losses
import numpy as np
import os
import h5py

file_dict = '/home/lukemarkham1383/trainEnvironment/npArrays/'    # Change this
for k in range(50):     # 50 means training for 50 epochs
    img_measure_file = 'Augment001-005PatientNS_Original.npy'
    bm_measure_file = 'Augment001-005PatientNS_Binary.npy'
    img_measure = np.load(os.path.join(file_dict, img_measure_file))
    bm_measure = np.load(os.path.join(file_dict, bm_measure_file)) / 255    # Converting to binary

    img_test_file = 'nonAugmentPatientNS_Original.npy'
    bm_test_file = 'nonAugmentPatientNS_Binary.npy'
    img_test = np.load(os.path.join(file_dict, img_test_file))
    bm_test = np.load(os.path.join(file_dict, bm_test_file)) / 255
    print(img_test.shape)
    print(img_measure.shape)
    testSplit = img_test.shape[0]/(img_test.shape[0]+img_measure.shape[0])
    img_train = np.concatenate((img_measure, img_test))
    bm_train = np.concatenate((bm_measure, bm_test))

    model_folder = '/home/lukemarkham1383/trainEnvironment/models'   # Change this
    model_list = os.listdir(model_folder)   # Checking if there is an existing model
    if model_list.__len__() == 0:   # Creating a new model if empty
        patch_rows = 512
        patch_cols = 512

        inputs = Input((patch_rows, patch_cols, 1))
        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
        conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)
        pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)

        conv6 = Conv2D(1024, (3, 3), activation='relu', padding='same')(pool5)
        conv6 = Conv2D(1024, (3, 3), activation='relu', padding='same')(conv6)
        '''
        The number of filters 32, 64, 128, ..., 512 are half of the original U-net. Please check this one in case
        I am wrong about it. It will be interesting to calculate the number of parameters of each layer, or look 
        them up in the summary, to understand the reason behind the choices.
        '''
        up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv5], axis=3)
        conv7 = Conv2D(512, (3, 3), activation='relu', padding='same')(up7)
        conv7 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv7)

        up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv4], axis=3)
        conv8 = Conv2D(256, (3, 3), activation='relu', padding='same')(up8)
        conv8 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv8)

        up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv3], axis=3)
        conv9 = Conv2D(128, (3, 3), activation='relu', padding='same')(up9)
        conv9 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv9)

        up10 = concatenate([UpSampling2D(size=(2, 2))(conv9), conv2], axis=3)
        conv10 = Conv2D(64, (3, 3), activation='relu', padding='same')(up10)
        conv10 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv10)

        up11 = concatenate([UpSampling2D(size=(2, 2))(conv10), conv1], axis=3)
        conv11 = Conv2D(32, (3, 3), activation='relu', padding='same')(up11)
        conv11 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv11)

        conv12 = Conv2D(1, (1, 1), activation='sigmoid')(conv11)

        model = Model(inputs=[inputs], outputs=[conv12])    # this Model function builds a functional API

        model.compile(optimizer=Adam(lr=1e-3), loss=losses.binary_crossentropy)     # The learning rate of optimizer is
        # very important, if you see your training profile diverge, it is probably caused by a big learning rate. You
        # can change the optimizer though, and there are ones which does not require a fix learning rate. Feel free to
        # explore.

        epoch_number = 0

    else:
        #model_file = model_list[-1]
	currentMax = 0;
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
        model.compile(optimizer=Adam(lr=1e-3), loss=losses.binary_crossentropy)


    epoch_number = epoch_number + 1
    model_check_file = os.path.join(model_folder, 'weights.{epoch:02d}-{loss:.2f}.h5')

    model_checkpoint = ModelCheckpoint(model_check_file, monitor='val_loss', save_best_only=False)

    history = model.fit(img_train, bm_train, batch_size=8, initial_epoch=epoch_number, epochs=epoch_number + 1,
                        verbose=1, shuffle=True, validation_split=0.2,
                        callbacks=[model_checkpoint])
    # You can find details about fit on keras website, just a little reminder here: validation_split starts before
    # shuffle, so don't worry if shuffle would blur both training and validating data. If OOM (out of memory appears),
    # try to make batch_size smaller.

    img_train = np.ndarray((0, 512, 512, 1), dtype='float32')
    bm_train = np.ndarray((0, 512, 512, 1), dtype='float32')
