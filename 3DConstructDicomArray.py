import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import scipy
from scipy import misc
import math
from imageProcessingFuntions import *

arrayDirectory = '/home/lukemarkham1383/trainEnvironment/npArrays/validation/'
imageDirectory = '/home/lukemarkham1383/trainEnvironment/Regent_RR/croppedDicoms/'

patientID = 'RR'
imageType = 'Original'
nonAugmentedVersion = True
binNum = 1

fileList = sorted(os.listdir(imageDirectory))
imgTotal = len(fileList)
totalCounter = 0
maxSliceNum = findLargestNumberInFolder(fileList)

npImageArray = np.ndarray((binNum*maxSliceNum, 5, 256, 256, 1), dtype='float32')

print('Loop starting')
for filename in fileList:

    print(filename)
    split1 = filename.split(imageType)
    if nonAugmentedVersion == True:
        split2 = split1[0].split('NonAugment')
	augNum = 777777777777777
    elif nonAugmentedVersion == False:
        split2 = split1[0].split('Augment')
        augNum = int(split2[1])
    split3 = split1[1].split('Patient')
    sliceNum = int(split3[0])
    if nonAugmentedVersion == False:
        arrayIndex = int(sliceNum - 1 + (augNum-1-((math.floor((augNum-1)/binNum))*binNum))*maxSliceNum)
    elif nonAugmentedVersion == True:
        arrayIndex = totalCounter

    image = misc.imread(imageDirectory + filename, flatten=True)

    if sliceNum > 4 and sliceNum < maxSliceNum - 3:
        #assign to this index
        npImageArray[arrayIndex, 2, :, :, 0] = image

        #assign to previous indexes
        npImageArray[arrayIndex-2, 3, :, :, 0] = image
        npImageArray[arrayIndex-4, 4, :, :, 0] = image

        #assign to future indexes
        npImageArray[arrayIndex+2, 1, :, :, 0] = image
        npImageArray[arrayIndex+4, 0, :, :, 0] = image

    elif sliceNum > 2 and sliceNum < 5: #gets slices 3 and 4
        #assign to this index
        npImageArray[arrayIndex, 2, :, :, 0] = image
        npImageArray[arrayIndex, 1, :, :, 0] = image
        npImageArray[arrayIndex, 0, :, :, 0] = image #this is done for contingency

        #assign to previous indexes
        npImageArray[arrayIndex - 2, 3, :, :, 0] = image

        # assign to future indexes
        npImageArray[arrayIndex + 2, 1, :, :, 0] = image
        npImageArray[arrayIndex + 4, 0, :, :, 0] = image

    elif sliceNum < 2: #gets slices 1 and 2
        # assign to this index
        npImageArray[arrayIndex, 2, :, :, 0] = image
        npImageArray[arrayIndex, 1, :, :, 0] = image
        npImageArray[arrayIndex, 0, :, :, 0] = image  # this is necessary

        # assign to future indexes
        npImageArray[arrayIndex + 2, 1, :, :, 0] = image
        npImageArray[arrayIndex + 4, 0, :, :, 0] = image
    elif sliceNum > maxSliceNum - 5 and sliceNum < maxSliceNum - 1: #gets slices which are 3rd and 4th from the end
        # assign to this index
        npImageArray[arrayIndex, 2, :, :, 0] = image
        npImageArray[arrayIndex, 3, :, :, 0] = image
        npImageArray[arrayIndex, 4, :, :, 0] = image  # this is done for contingency

        # assign to previous indexes
        npImageArray[arrayIndex - 2, 3, :, :, 0] = image
        npImageArray[arrayIndex - 4, 4, :, :, 0] = image

        # assign to future indexes
        npImageArray[arrayIndex + 2, 1, :, :, 0] = image

    elif sliceNum > maxSliceNum - 3: #gets the end and the one before it
        #assigns to this index
        npImageArray[arrayIndex, 2, :, :, 0] = image
        npImageArray[arrayIndex, 3, :, :, 0] = image
        npImageArray[arrayIndex, 4, :, :, 0] = image #this is needed

        #assigns to prevous indexes
        npImageArray[arrayIndex - 2, 3, :, :, 0] = image
        npImageArray[arrayIndex - 4, 4, :, :, 0] = image

    totalCounter = totalCounter + 1

    if ((augNum%binNum == 0) or (nonAugmentedVersion == True)) and (sliceNum == maxSliceNum):
	if (nonAugmentedVersion == True):
		np.save(arrayDirectory + '3DNonAugment' + 'Patient' + patientID + '_' + imageType + '.npy', npImageArray)
	else:
        	np.save(arrayDirectory + '3DAugment' + "%03d" % (augNum-binNum+1) + '-' + "%03d" % (augNum) + 'Patient' + patientID + '_' + imageType + '.npy', npImageArray)
	print('Saved one at augNum ' + str(augNum))
