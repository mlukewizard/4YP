from myFunctions import *

patientList = ['NS', 'PB', 'PS', 'RR', 'DC']
augmentedList = [True, False, False, False, False]

myBinNum = 1
myReturnArray = False
mySaveArray = True

for iteration in range(len(patientList)):
    myPatientID = patientList[iteration]
    myNonAugmentedVersion = augmentedList[iteration]

    print('Constructing dicom arrays for patient ' + myPatientID)

    if myNonAugmentedVersion ==  True:
        myArrayDirectory = '/home/lukemarkham1383/trainEnvironment/npArrays/validation/'
    else:
        myArrayDirectory = '/home/lukemarkham1383/trainEnvironment/npArrays/training/'
    myImageDirectory = '/home/lukemarkham1383/trainEnvironment/Regent_' + myPatientID + '/croppedDicoms/'

    Construct3DDicomArray(myImageDirectory, myArrayDirectory, myPatientID, myNonAugmentedVersion, myBinNum, myReturnArray, mySaveArray)