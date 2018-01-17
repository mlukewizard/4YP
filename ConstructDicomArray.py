from myFunctions import *
import os
import numpy as np
import matplotlib.pyplot as plt
from uuid import getnode as get_mac
mac = get_mac()
boxSize = 144
twoDVersion = False

patientList = ['NS', 'PB', 'PS', 'RR', 'DC']
augmentedList = [True, False, False, False, False]

for myPatientID, myNonAugmentedVersion in zip(patientList, augmentedList):
    print('Constructing dicom arrays for patient ' + myPatientID)

    if mac != 176507742233701:
        if myNonAugmentedVersion:
            myArrayDirectory = '/home/lukemarkham1383/trainEnvironment/npArrays/validation/'
        else:
            myArrayDirectory = '/home/lukemarkham1383/trainEnvironment/npArrays/training/'
        myImageDirectory = '/home/lukemarkham1383/trainEnvironment/Regent_' + myPatientID + '/croppedDicoms/'
    else:
        myArrayDirectory = 'C:\\Users\\Luke\\Documents\\sharedFolder\\4YP\\npArrays\\'
        myImageDirectory = 'C:\\Users\\Luke\\Documents\\sharedFolder\\4YP\\Images\\Regent_' + myPatientID + '\\postAugmentation\\croppedDicoms\\'

    fileList = sorted(os.listdir(myImageDirectory))
    maxSliceNum = findLargestNumberInFolder(fileList)

    maxAugNum = 1 if myNonAugmentedVersion else 5

    if not twoDVersion:
        npImageArray = np.ndarray((maxSliceNum, 5, boxSize, boxSize, 1), dtype='float32')
    else:
        npImageArray = np.ndarray((maxSliceNum, boxSize, boxSize, 1), dtype='float32')

    for augNum in range(1, maxAugNum + 1):
        if not myNonAugmentedVersion:
            workingFileList = [x for x in fileList if 'Augment0' + str(augNum) in x]
        else:
            workingFileList = fileList
        for j in range(maxSliceNum):
            if not twoDVersion:
                npImageArray[j, :, :, :, :] = ConstructArraySlice(workingFileList, myImageDirectory, j, boxSize, twoDVersion=True)
                #saveSlice(npImageArray[j, :, :, :, :], showFig = True)
            else:
                npImageArray[j, :, :, :] = ConstructArraySlice(workingFileList, myImageDirectory, j, boxSize, twoDVersion=True)
                #saveSlice(npImageArray[j, :, :, :], showFig = True)


        # Saves down the numpy array
        if myNonAugmentedVersion:
            if not twoDVersion:
                np.save(myArrayDirectory + '3DNonAugment' + 'Patient' + myPatientID + '_' + 'dicom' + '.npy', npImageArray)
            else:
                np.save(myArrayDirectory + '2DNonAugment' + 'Patient' + myPatientID + '_' + 'dicom' + '.npy', npImageArray)
        else:
            if not twoDVersion:
                np.save(myArrayDirectory + '3DAugment0' + str(augNum) + 'Patient' + myPatientID + '_' + 'dicom' + '.npy', npImageArray)
            else:
                np.save(myArrayDirectory + '2DAugment0' + str(augNum) + 'Patient' + myPatientID + '_' + 'dicom' + '.npy', npImageArray)
        print('Saved one at augNum ' + str(augNum))
