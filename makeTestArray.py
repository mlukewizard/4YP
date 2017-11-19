import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import scipy
from scipy import misc
from  __builtin__ import any as b_any

imageDirectory = '/media/sf_sharedFolder/Images/39894NS/PostAugmentation/augmentedInnerOriginals/Augment006Original264PatientNS.png'
patientID = 'NS'

npImageArray = np.ndarray((1, 512, 512, 1), dtype='float32')

image = misc.imread(imageDirectory)

plt.imshow(image)
plt.show()

npImageArray[0, :, :, 0] = image

np.save('/media/sf_sharedFolder/npArrays/39894NS/' + 'Augment006Original264PatientNSTestCase' + '.npy', npImageArray)