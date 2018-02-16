from __future__ import print_function
from __future__ import division
from keras.callbacks import ModelCheckpoint
from keras.models import Model, load_model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, ConvLSTM2D, LSTM, TimeDistributed, Bidirectional, Dropout, BatchNormalization, Activation
from keras.optimizers import Adam
from keras import losses
from keras import regularizers
import numpy as np
import os
import sys
import h5py
from keras import optimizers
from keras import backend as K
from random import *

def my3DModel(boxSize):
    inputs = Input((5, boxSize, boxSize, 1))

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
    return model

def my3DModelDoubled(boxSize):
    inputs = Input((5, boxSize, boxSize, 1))

    conv1 = TimeDistributed(BatchNormalization())(inputs)
    conv1 = TimeDistributed(Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01)))(conv1)
    conv1 = TimeDistributed(BatchNormalization())(conv1)
    conv1 = TimeDistributed(Activation('relu'))(conv1)
    conv1 = TimeDistributed(Dropout(0.3))(conv1)
    conv1 = TimeDistributed(Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01)))(conv1)
    conv1 = TimeDistributed(BatchNormalization())(conv1)
    conv1 = TimeDistributed(Activation('relu'))(conv1)
    conv1 = TimeDistributed(Dropout(0.3))(conv1)
    pool1 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv1)
    pool1 = TimeDistributed(Dropout(0.3))(pool1)

    conv2 = TimeDistributed(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01)))(pool1)
    conv2 = TimeDistributed(BatchNormalization())(conv2)
    conv2 = TimeDistributed(Activation('relu'))(conv2)
    conv2 = TimeDistributed(Dropout(0.3))(conv2)
    conv2 = TimeDistributed(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01)))(pool1)
    conv2 = TimeDistributed(BatchNormalization())(conv2)
    conv2 = TimeDistributed(Activation('relu'))(conv2)
    conv2 = TimeDistributed(Dropout(0.3))(conv2)
    pool2 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv2)
    pool2 = TimeDistributed(Dropout(0.3))(pool2)

    conv3 = TimeDistributed(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01)))(pool2)
    conv3 = TimeDistributed(BatchNormalization())(conv3)
    conv3 = TimeDistributed(Activation('relu'))(conv3)
    conv3 = TimeDistributed(Dropout(0.3))(conv3)
    conv3 = TimeDistributed(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01)))(pool2)
    conv3 = TimeDistributed(BatchNormalization())(conv3)
    conv3 = TimeDistributed(Activation('relu'))(conv3)
    conv3 = TimeDistributed(Dropout(0.3))(conv3)
    pool3 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv3)
    pool3 = TimeDistributed(Dropout(0.3))(pool3)

    conv4 = TimeDistributed(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01)))(pool3)
    conv4 = TimeDistributed(BatchNormalization())(conv4)
    conv4 = TimeDistributed(Activation('relu'))(conv4)
    conv4 = TimeDistributed(Dropout(0.3))(conv4)
    conv4 = TimeDistributed(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01)))(pool3)
    conv4 = TimeDistributed(BatchNormalization())(conv4)
    conv4 = TimeDistributed(Activation('relu'))(conv4)
    conv4 = TimeDistributed(Dropout(0.3))(conv4)
    pool4 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv4)
    pool4 = TimeDistributed(Dropout(0.3))(pool4)

    myLSTM = Bidirectional(ConvLSTM2D(512, (3, 3), padding='same', return_sequences=True, kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01)))(pool4)
    myLSTM = TimeDistributed(BatchNormalization())(myLSTM)
    myLSTM = TimeDistributed(Activation('relu'))(myLSTM)

    up6 = concatenate([TimeDistributed(UpSampling2D(size=(2, 2)))(myLSTM), conv4], axis=4)
    conv6 = TimeDistributed(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01)))(up6)
    conv6 = TimeDistributed(BatchNormalization())(conv6)
    conv6 = TimeDistributed(Activation('relu'))(conv6)
    conv6 = TimeDistributed(Dropout(0.3))(conv6)
    conv6 = TimeDistributed(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01)))(up6)
    conv6 = TimeDistributed(BatchNormalization())(conv6)
    conv6 = TimeDistributed(Activation('relu'))(conv6)
    conv6 = TimeDistributed(Dropout(0.3))(conv6)

    up7 = concatenate([TimeDistributed(UpSampling2D(size=(2, 2)))(conv6), conv3], axis=4)
    conv7 = TimeDistributed(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01)))(up7)
    conv7 = TimeDistributed(BatchNormalization())(conv7)
    conv7 = TimeDistributed(Activation('relu'))(conv7)
    conv7 = TimeDistributed(Dropout(0.3))(conv7)
    conv7 = TimeDistributed(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01)))(up7)
    conv7 = TimeDistributed(BatchNormalization())(conv7)
    conv7 = TimeDistributed(Activation('relu'))(conv7)
    conv7 = TimeDistributed(Dropout(0.3))(conv7)

    up8 = concatenate([TimeDistributed(UpSampling2D(size=(2, 2)))(conv7), conv2], axis=4)
    conv8 = TimeDistributed(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01)))(up8)
    conv8 = TimeDistributed(BatchNormalization())(conv8)
    conv8 = TimeDistributed(Activation('relu'))(conv8)
    conv8 = TimeDistributed(Dropout(0.3))(conv8)
    conv8 = TimeDistributed(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01)))(up8)
    conv8 = TimeDistributed(BatchNormalization())(conv8)
    conv8 = TimeDistributed(Activation('relu'))(conv8)
    conv8 = TimeDistributed(Dropout(0.3))(conv8)

    up9 = concatenate([TimeDistributed(UpSampling2D(size=(2, 2)))(conv8), conv1], axis=4)
    conv9 = TimeDistributed(Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01)))(up9)
    conv9 = TimeDistributed(BatchNormalization())(conv9)
    conv9 = TimeDistributed(Activation('relu'))(conv9)
    conv9 = TimeDistributed(Dropout(0.3))(conv9)
    conv9 = TimeDistributed(Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01)))(up9)
    conv9 = TimeDistributed(BatchNormalization())(conv9)
    conv9 = TimeDistributed(Activation('relu'))(conv9)
    conv9 = TimeDistributed(Dropout(0.3))(conv9)

    conv10 = TimeDistributed(Conv2D(2, (1, 1)))(conv9)
    conv10 = TimeDistributed(BatchNormalization())(conv10)
    conv10 = TimeDistributed(Activation('sigmoid'))(conv10)

    model = Model(inputs=[inputs], outputs=[conv10])
    return model

def my2DModel(boxSize):
    inputs = Input((boxSize, boxSize, 1))

    conv1 = BatchNormalization()(inputs)
    conv1 = Conv2D(32, (3, 3), padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = Dropout(0.2)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    pool1 = Dropout(0.2)(pool1)

    conv2 = Conv2D(64, (3, 3), padding='same')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    conv2 = Dropout(0.2)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    pool2 = Dropout(0.2)(pool2)

    conv3 = Conv2D(128, (3, 3), padding='same')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)
    conv3 = Dropout(0.2)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    pool3 = Dropout(0.2)(pool3)

    conv4 = Conv2D(256, (3, 3), padding='same')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation('relu')(conv4)
    conv4 = Dropout(0.2)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    pool4 = Dropout(0.2)(pool4)

    conv5 = Conv2D(256, (3, 3), padding='same')(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation('relu')(conv5)
    conv5 = Dropout(0.2)(conv5)

    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), padding='same')(up6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Activation('relu')(conv6)
    conv6 = Dropout(0.2)(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), padding='same')(up7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Activation('relu')(conv7)
    conv7 = Dropout(0.2)(conv7)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), padding='same')(up8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Activation('relu')(conv8)
    conv8 = Dropout(0.2)(conv8)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), padding='same')(up9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Activation('relu')(conv9)
    conv9 = Dropout(0.2)(conv9)

    conv10 = Conv2D(2, (1, 1))(conv9)
    conv10 = BatchNormalization()(conv10)
    conv10 = Activation('sigmoid')(conv10)

    model = Model(inputs=[inputs], outputs=[conv10])
    return model
