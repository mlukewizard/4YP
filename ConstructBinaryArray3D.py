from myFunctions import *

patientList = ['NS', 'PB', 'PS', 'RR', 'DC']
augmentedList = [True, False, False, False, False]

myBinNum = 1
myReturnArray = False
mySaveArray = True

for iteration in range(len(patientList)):
    myPatientID = patientList[iteration]
    myNonAugmentedVersion = augmentedList[iteration]

    print('Constructing binary arrays for patient ' + myPatientID)

    if myNonAugmentedVersion ==  True:
        myArrayDirectory = '/home/lukemarkham1383/trainEnvironment/npArrays/validation/'
    else:
        myArrayDirectory = '/home/lukemarkham1383/trainEnvironment/npArrays/training/'
    myInnerImageDirectory = '/home/lukemarkham1383/trainEnvironment/Regent_' + myPatientID + '/innerAugmented/'
    myOuterImageDirectory = '/home/lukemarkham1383/trainEnvironment/Regent_' + myPatientID + '/outerAugmented/'

    Construct3DBinaryArray(myInnerImageDirectory, myOuterImageDirectory, myArrayDirectory, myPatientID, myNonAugmentedVersion, myBinNum, myReturnArray, mySaveArray)
