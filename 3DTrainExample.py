from __future__ import print_function
from keras.callbacks import ModelCheckpoint
from keras.models import Model, load_model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, ConvLSTM2D, LSTM, TimeDistributed, Bidirectional, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras import losses
import numpy as np
import os
import h5py
from keras import optimizers
from keras import backend as K


def my_loss(y_true, y_pred):
    smooth = 0
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice_coeff = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return (-1)*dice_coeff
#    return K.mean(K.binary_crossentropy(y_true[:,2,:,:,:], y_pred[:,2,:,:,:], axis=-1)

file_dict = '/home/lukemarkham1383/trainEnvironment/npArrays/'  # Change this
for k in range(10):  # 50 means training for 50 epochs
    img_measure_file = '3DAugment001-002PatientNS_Original.npy'
    bm_measure_file = '3DAugment001-002PatientNS_Binary.npy'
    img_train = np.load(os.path.join(file_dict, img_measure_file))
    bm_train = np.load(os.path.join(file_dict, bm_measure_file)) / 255  # Converting to binary

    # img_test_file = 'nonAugmentPatientNS_Original.npy'
    # bm_test_file = 'nonAugmentPatientNS_Binary.npy'
    # img_test = np.load(os.path.join(file_dict, img_test_file))
    # bm_test = np.load(os.path.join(file_dict, bm_test_file)) / 255

    # testSplit = img_test.shape[0]/(img_test.shape[0]+img_measure.shape[0])
    #img_train = img_measure
    # np.concatenate((img_measure, img_test))
    #bm_train = bm_measure
    # np.concatenate((bm_measure, bm_test))

    model_folder = '/home/lukemarkham1383/trainEnvironment/models'  # Change this
    model_list = os.listdir(model_folder)  # Checking if there is an existing model
    if model_list.__len__() == 0:  # Creating a new model if empty

        inputs = Input((5, 256, 256, 1))

        conv1 = TimeDistributed(BatchNormalization())(inputs)
        conv1 = TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same'))(conv1)
        conv1 = TimeDistributed(BatchNormalization())(conv1)
        conv1 = TimeDistributed(Dropout(0.5))(conv1)
        conv1 = TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same'))(conv1)
        pool1 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv1)

        conv2 = TimeDistributed(BatchNormalization())(pool1)
        conv2 = TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same'))(conv2)
        conv2 = TimeDistributed(BatchNormalization())(conv2)
        conv2 = TimeDistributed(Dropout(0.5))(conv2)
        conv2 = TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same'))(conv2)
        pool2 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv2)

        conv3 = TimeDistributed(BatchNormalization())(pool2)
        conv3 = TimeDistributed(Conv2D(128, (3, 3), activation='relu', padding='same'))(conv3)
        conv3 = TimeDistributed(BatchNormalization())(conv3)
        conv3 = TimeDistributed(Dropout(0.5))(conv3)
        conv3 = TimeDistributed(Conv2D(128, (3, 3), activation='relu', padding='same'))(conv3)
        pool3 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv3)

        conv4 = TimeDistributed(BatchNormalization())(pool3)
        conv4 = TimeDistributed(Conv2D(256, (3, 3), activation='relu', padding='same'))(conv4)
        conv4 = TimeDistributed(BatchNormalization())(conv4)
        conv4 = TimeDistributed(Dropout(0.5))(conv4)
        conv4 = TimeDistributed(Conv2D(256, (3, 3), activation='relu', padding='same'))(conv4)
        pool4 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv4)

        myLSTM = Bidirectional(ConvLSTM2D(512, (3, 3), activation='relu', padding='same', return_sequences=True))(pool4)

        up6 = concatenate([TimeDistributed(UpSampling2D(size=(2, 2)))(myLSTM), conv4], axis=4)
        conv6 = TimeDistributed(BatchNormalization())(up6)
        conv6 = TimeDistributed(Conv2D(256, (3, 3), activation='relu', padding='same'))(conv6)
        conv6 = TimeDistributed(BatchNormalization())(conv6)
        conv6 = TimeDistributed(Dropout(0.5))(conv6)
        conv6 = TimeDistributed(Conv2D(256, (3, 3), activation='relu', padding='same'))(conv6)

        up7 = concatenate([TimeDistributed(UpSampling2D(size=(2, 2)))(conv6), conv3], axis=4)
        conv7 = TimeDistributed(BatchNormalization())(up7)
        conv7 = TimeDistributed(Conv2D(128, (3, 3), activation='relu', padding='same'))(conv7)
        conv7 = TimeDistributed(BatchNormalization())(conv7)
        conv7 = TimeDistributed(Dropout(0.5))(conv7)
        conv7 = TimeDistributed(Conv2D(128, (3, 3), activation='relu', padding='same'))(conv7)

        up8 = concatenate([TimeDistributed(UpSampling2D(size=(2, 2)))(conv7), conv2], axis=4)
        conv8 = TimeDistributed(BatchNormalization())(up8)
        conv8 = TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same'))(conv8)
        conv8 = TimeDistributed(BatchNormalization())(conv8)
        conv8 = TimeDistributed(Dropout(0.5))(conv8)
        conv8 = TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same'))(conv8)

        up9 = concatenate([TimeDistributed(UpSampling2D(size=(2, 2)))(conv8), conv1], axis=4)
        conv9 = TimeDistributed(BatchNormalization())(up9)
        conv9 = TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same'))(conv9)
        conv9 = TimeDistributed(BatchNormalization())(conv9)
        conv9 = TimeDistributed(Dropout(0.5))(conv9)
        conv9 = TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same'))(conv9)

        conv10 = TimeDistributed(BatchNormalization())(conv9)
        conv10 = TimeDistributed(Dropout(0.5))(conv10)
        conv10 = TimeDistributed(Conv2D(1, (1, 1), activation='sigmoid'))(conv10)

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
	
    model.summary()
    model.compile(optimizer=Adam(lr=1e-4), loss=my_loss)

    model_check_file = os.path.join(model_folder, 'weights.{epoch:02d}-{loss:.2f}.h5')

    model_checkpoint = ModelCheckpoint(model_check_file, monitor='val_loss', save_best_only=False)

    history = model.fit(img_train, bm_train, batch_size=4, initial_epoch=epoch_number, epochs=epoch_number + 10,
                        verbose=1, shuffle=True, validation_split=0.5,
                        callbacks=[model_checkpoint])
