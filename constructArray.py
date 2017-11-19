import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import scipy
from scipy import misc
import math

imageDirectory = '/home/lukemarkham1383/trainEnvironment/nonAugmentedInnerBinary/'
arrayDirectory = '/home/lukemarkham1383/trainEnvironment/npArrays/'
patientID = 'NS'
imageType = 'Binary'

fileList = sorted(os.listdir(imageDirectory))
imgTotal = len(fileList)
totalCounter = 0
maxSliceNum = 475
binNum = 1
nonAugmentedVersion = True

npImageArray = np.ndarray((binNum * maxSliceNum, 512, 512, 1), dtype='float32')

print('Loop starting')
for filename in fileList:

    split1 = filename.split(imageType)
    split2 = split1[0].split('Augment')
    augNum = int(split2[1])
    split3 = split1[1].split('Patient')
    sliceNum = int(split3[0])
    arrayIndex = int(sliceNum - 1 + (augNum-1-((math.floor((augNum-1)/binNum))*binNum))*maxSliceNum)
    image = misc.imread(imageDirectory + filename)
    #print(image[250,250])
    #plt.imshow(image)
    #plt.show()
    npImageArray[arrayIndex, :, :, 0] = image
    totalCounter = totalCounter + 1

    if (augNum%binNum == 0) and (sliceNum == maxSliceNum):
	print('Saved one at augNum ' + str(augNum))
	if (nonAugmentedVersion == True):
		np.save(arrayDirectory + 'nonAugment' + 'Patient' + patientID + '_' + imageType + '.npy', npImageArray)
	else:
        	np.save(arrayDirectory + 'Augment' + "%03d" % (augNum-4) + '-' + "%03d" % (augNum) + 'Patient' + patientID + '_' + imageType + '.npy', npImageArray)

