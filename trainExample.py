import os
from myFunctions import trainModel
import gc
import subprocess

#Program inputs
twoDVersion = False
patientList = ['PS', 'PB', 'RR', 'NS', 'AA', 'AD']
trainingArrayDepth = 300
augmentationsInTrainingArray = len(patientList)
boxSize = 256

#Defining file paths
validationArrayPath = '/home/lukemarkham1383/trainEnvironment/npArrays/validation/'
trainingArrayPath = '/home/lukemarkham1383/trainEnvironment/npArrays/training/'
model_folder = '/home/lukemarkham1383/trainEnvironment/models/'

if not twoDVersion:
    img_test_files = ['3DNonAugmentPatientDC_dicom.npy', '3DNonAugmentPatientAG_dicom.npy']
    bm_test_files = ['3DNonAugmentPatientDC_binary.npy', '3DNonAugmentPatientAG_binary.npy']
else:
    img_test_file = '2DNonAugmentPatientNS_dicom.npy'
    bm_test_file = '2DNonAugmentPatientNS_binary.npy'

dicomFileList = []
for k in range(100):
    dicomFileList = trainModel(patientList, trainingArrayDepth, twoDVersion, boxSize, dicomFileList, trainingArrayPath, validationArrayPath, model_folder, img_test_files, bm_test_files)
    gc.collect()
    subprocess.call('/home/lukemarkham1383/gdrive-linux-x64 upload /home/lukemarkham1383/trainEnvironment/4YP_Python/programlog.txt', shell=True)
