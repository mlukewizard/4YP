from myFunctions import *

myArrayDirectory = '/home/lukemarkham1383/trainEnvironment/npArrays/validation/'
myImageDirectory = '/home/lukemarkham1383/trainEnvironment/Regent_RR/croppedDicoms/'

myPatientID = 'RR'
myNonAugmentedVersion = True
myBinNum = 1
myReturnArray = False
mySaveArray = True

Construct3DDicomArray(myImageDirectory, myArrayDirectory, myPatientID, myNonAugmentedVersion, myBinNum, myReturnArray, mySaveArray)