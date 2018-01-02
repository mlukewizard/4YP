from myFunctions import *

patientList = ['NS', 'PB', 'PS', 'RR', 'DC']
augmentedList = [False, True, True, True, True]

myBinNum = 1
myReturnArray = False
mySaveArray = True

for iteration in range(len(patientList)):
    myPatientID = patientList[iteration]
    myNonAugmentedVersion = augmentedList[iteration]

    myArrayDirectory = '/home/lukemarkham1383/trainEnvironment/npArrays/validation/'
    myImageDirectory = '/home/lukemarkham1383/trainEnvironment/Regent_' + myPatientID + '/croppedDicoms/'

    Construct3DDicomArray(myImageDirectory, myArrayDirectory, myPatientID, myNonAugmentedVersion, myBinNum, myReturnArray, mySaveArray)