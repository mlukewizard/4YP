import os
from myFunctions import trainModel

#Program inputs
twoDVersion = False
patientList = ['PS', 'PB', 'RR', 'DC']
trainingArrayDepth = 300 
augmentationsInTrainingArray = len(patientList)
boxSize = 256

#Defining file paths
validationArrayPath = '/home/lukemarkham1383/trainEnvironment/npArrays/validation/'
trainingArrayPath = '/home/lukemarkham1383/trainEnvironment/npArrays/training/'
model_folder = '/home/lukemarkham1383/trainEnvironment/models/'

if not twoDVersion:
    img_test_file = '3DNonAugmentPatientNS_dicom.npy'
    bm_test_file = '3DNonAugmentPatientNS_binary.npy'
else:
    img_test_file = '2DNonAugmentPatientNS_dicom.npy'
    bm_test_file = '2DNonAugmentPatientNS_binary.npy'

dicomFileList = []
for k in range(100):
    dicomFileList = trainModel(patientList, trainingArrayDepth, twoDVersion, boxSize, dicomFileList, trainingArrayPath, validationArrayPath, model_folder, img_test_file, bm_test_file)