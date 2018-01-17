from myFunctions import *
import os
import numpy as np
import matplotlib.pyplot as plt
from uuid import getnode as get_mac
mac = get_mac()
boxSize = 150

patientList = ['NS', 'PB', 'PS', 'RR', 'DC']
augmentedList = [True, False, False, False, False]

for myPatientID, myNonAugmentedVersion in zip(patientList, augmentedList):
    print('Constructing binary arrays for patient ' + myPatientID)

    if mac != 176507742233701:
        if myNonAugmentedVersion ==  True:
            myArrayDirectory = '/home/lukemarkham1383/trainEnvironment/npArrays/validation/'
        else:
            myArrayDirectory = '/home/lukemarkham1383/trainEnvironment/npArrays/training/'
        myInnerImageDirectory = '/home/lukemarkham1383/trainEnvironment/Regent_' + myPatientID + '/innerAugmented/'
        myOuterImageDirectory = '/home/lukemarkham1383/trainEnvironment/Regent_' + myPatientID + '/outerAugmented/'
    else:
        myArrayDirectory = 'C:\\Users\\Luke\\Documents\\sharedFolder\\4YP\\npArrays\\'
        myInnerImageDirectory = 'C:\\Users\\Luke\\Documents\\sharedFolder\\4YP\\Images\\Regent_' + myPatientID + '\\postAugmentation\\innerAugmented\\'
        myOuterImageDirectory = 'C:\\Users\\Luke\\Documents\\sharedFolder\\4YP\\Images\\Regent_' + myPatientID + '\\postAugmentation\\outerAugmented\\'

    innerFileList = sorted(os.listdir(myInnerImageDirectory))
    outerFileList = sorted(os.listdir(myOuterImageDirectory))
    maxSliceNum = findLargestNumberInFolder(innerFileList)
    npImageArray = np.ndarray((maxSliceNum, 5, boxSize, boxSize, 2), dtype='float32')

    maxAugNum = 1 if myNonAugmentedVersion else 5

    for augNum in range(1, maxAugNum+1):
        if not myNonAugmentedVersion:
            workingInnerFileList = [x for x in innerFileList if 'Augment0' + str(augNum) in x]
            workingOuterFileList = [x for x in outerFileList if 'Augment0' + str(augNum) in x]
        else:
            workingInnerFileList = innerFileList
            workingOuterFileList = outerFileList

        for j in range(maxSliceNum):
            npImageArray[j, :, :, :, :] = ConstructArraySlice(workingInnerFileList, myInnerImageDirectory, workingOuterFileList, myOuterImageDirectory, j, None, boxSize)
            #saveSlice(npImageArray[j, :, :, :, :], None)

        if myNonAugmentedVersion:
            np.save(myArrayDirectory + '3DNonAugment' + 'Patient' + myPatientID + '_' + 'binary' + '.npy', npImageArray)
        else:
            np.save(
                myArrayDirectory + '3DAugment0' + str(augNum) + 'Patient' + myPatientID + '_' + 'binary' + '.npy', npImageArray)
        print('Saved one at augNum ' + str(augNum))
