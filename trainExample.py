import os
from myFunctions import trainModel
import gc
import subprocess

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
    gc.collect()
    #subprocess.call('sudo kill -9 $(lsof /dev/nvidia*| grep python | awk \'{print $2; exit }\')', shell=True)
    subprocess.call('/home/lukemarkham1383/gdrive-linux-x64 upload /home/lukemarkham1383/trainEnvironment/4YP_Python/programlog.txt', shell=True)
