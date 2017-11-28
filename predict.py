from __future__ import print_function
from keras.callbacks import ModelCheckpoint
from keras.models import Model, load_model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from keras.optimizers import Adam
from keras import losses
import numpy as np
import h5py
import matplotlib
import matplotlib.pyplot as plt


model_file = '/media/sf_sharedFolder/Models/aaa/20thNov/weights.07-0.01.h5'

x = np.load('/media/sf_sharedFolder/npArrays/39894NS/Augment001-002PatientNS_Original.npy')

fig = plt.figure()
model = load_model(model_file)
npImageArray = np.ndarray((1, 256, 256, 1), dtype='float32')

for i in range(200, 400, 50):
    x1 = x[i, :, :, 0]
    npImageArray[0, :, :, 0] = x[i,:,:,0]

    y = model.predict(npImageArray)
    y1 = y[0, :, :, 0]

    plt.subplot(121)
    plt.imshow(x1, cmap='gray')
    plt.subplot(122)
    plt.imshow(y1, cmap='gray')
    plt.show()
