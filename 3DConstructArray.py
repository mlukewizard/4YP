import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import scipy
from scipy import misc
import math

imageDirectory = '/home/lukemarkham1383/trainEnvironment/augmentedInnerBinary/'
arrayDirectory = '/home/lukemarkham1383/trainEnvironment/npArrays/'
patientID = 'NS'
imageType = 'Binary'

fileList = sorted(os.listdir(imageDirectory))
imgTotal = len(fileList)
totalCounter = 0
maxSliceNum = 285
binNum = 2
nonAugmentedVersion = False

npImageArray = np.ndarray((binNum*maxSliceNum, 5, 256, 256, 1), dtype='float32')

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

    if (augNum%binNum == 0) and (sliceNum == maxSliceNum):
	print('Saved one at augNum ' + str(augNum))
	if (nonAugmentedVersion == True):
		np.save(arrayDirectory + '3DnonAugment' + 'Patient' + patientID + '_' + imageType + '.npy', npImageArray)
	else:
        	np.save(arrayDirectory + '3DAugment' + "%03d" % (augNum-binNum+1) + '-' + "%03d" % (augNum) + 'Patient' + patientID + '_' + imageType + '.npy', npImageArray)


