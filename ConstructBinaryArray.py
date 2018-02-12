from myFunctions import *
import os
import numpy as np
import matplotlib.pyplot as plt
from uuid import getnode as get_mac
mac = get_mac()
boxSize = 256
twoDVersion = False
augNum = 5

patientList = ['AA', 'AD', 'RR', 'NS', 'PB', 'PS', 'DC']
augmentedList = [False, False, False, False, False, False, True]

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

    maxAugNum = 1 if myNonAugmentedVersion else 5

    if not twoDVersion:
        npImageArray = np.ndarray((maxSliceNum, 5, boxSize, boxSize, 2), dtype='float32')
    else:
        npImageArray = np.ndarray((maxSliceNum, boxSize, boxSize, 2), dtype='float32')

    for augNum in range(1, maxAugNum+1):
        if not myNonAugmentedVersion:
            workingInnerFileList = [x for x in innerFileList if ('Augment' + '%02d' % (augNum,)) in x]
            workingOuterFileList = [x for x in outerFileList if ('Augment' + '%02d' % (augNum,)) in x]
        else:
            workingInnerFileList = innerFileList
            workingOuterFileList = outerFileList

        for j in range(maxSliceNum):
            if not twoDVersion:
                npImageArray[j, :, :, :, :] = ConstructArraySlice(workingInnerFileList, myInnerImageDirectory,  j, boxSize, inputFolder2 = workingOuterFileList, inputFolder2Dir = myOuterImageDirectory)
                #saveSlice(npImageArray[j, :, :, :, :], showFig = True)
            else:
                npImageArray[j, :, :, :] = ConstructArraySlice(workingInnerFileList, myInnerImageDirectory,  j, boxSize, inputFolder2 = workingOuterFileList, inputFolder2Dir = myOuterImageDirectory, twoDVersion = True)
                #saveSlice(npImageArray[j, :, :, :], showFig = True)

        if myNonAugmentedVersion:
            if not twoDVersion:
                np.save(myArrayDirectory + '3DNonAugment' + 'Patient' + myPatientID + '_' + 'binary' + '.npy', npImageArray)
            else:
                np.save(myArrayDirectory + '2DNonAugment' + 'Patient' + myPatientID + '_' + 'binary' + '.npy', npImageArray)
        else:
            if not twoDVersion:
                np.save(myArrayDirectory + ('3DAugment' + '%02d' % (augNum,)) + 'Patient' + myPatientID + '_' + 'binary' + '.npy', npImageArray)
            else:
                np.save(myArrayDirectory + ('2DAugment' + '%02d' % (augNum,)) + 'Patient' + myPatientID + '_' + 'binary' + '.npy', npImageArray)
        print('Saved one at augNum ' + str(augNum))
