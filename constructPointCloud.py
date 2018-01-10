import os
import csv
import numpy as np
from myFunctions import *
import scipy.misc as misc
from scipy import ndimage
import copy


patientList = ['NS', 'DC', 'PB', 'PS', 'RR']
for PatientID in patientList:
    print('Extracting patient ' + PatientID)
    innerBinaryDir = 'C:\\Users\\Luke\\Documents\\sharedFolder\\4YP\\Images\\Regent_'+PatientID+'\\preAugmentation\\innerBinary\\'
    CSVWriteDir = 'C:\\Users\\Luke\\Documents\\sharedFolder\\4YP\\pointClouds\\Regent_'+PatientID+'\\'
    if not os.path.exists(CSVWriteDir):
            os.mkdir(CSVWriteDir)
    outerBinaryDir = 'C:\\Users\\Luke\\Documents\\sharedFolder\\4YP\\Images\\Regent_'+PatientID+'\\preAugmentation\\outerBinary\\'

    # Extract the points of the aorta
    innerFileList = sorted(os.listdir(innerBinaryDir))
    outerFileList = sorted(os.listdir(outerBinaryDir))
    height = len(innerFileList)
    sampleImage = misc.imread(innerBinaryDir + innerFileList[0])
    innerPointCloud = np.zeros([height, sampleImage.shape[0], sampleImage.shape[1]], dtype=np.dtype(bool))
    innerImageHolder = np.zeros([sampleImage.shape[0], sampleImage.shape[1]], dtype=np.dtype(bool))
    outerPointCloud = np.zeros([height, sampleImage.shape[0], sampleImage.shape[1]], dtype=np.dtype(bool))
    outerImageHolder = np.zeros([sampleImage.shape[0], sampleImage.shape[1]], dtype=np.dtype(bool))
    adjustedOuterImageHolder = np.zeros([sampleImage.shape[0], sampleImage.shape[1]], dtype=np.dtype(bool))

    fileNum = -1
    innerPoint = np.zeros([2,1])
    for innerFileName in innerFileList:
        fileNum = fileNum + 1
        outerFileName = outerFileList[fileNum]
        innerImage = misc.imread(innerBinaryDir + innerFileName)
        innerImageHolder = getImagePerimeterPoints(innerImage)
        outerImage = misc.imread(outerBinaryDir + outerFileName)
        outerImageHolder = getImagePerimeterPoints(outerImage)
        outerPoints = np.array(np.where(np.isin(outerImageHolder, 1)))
        innerPoints = np.array(np.where(np.isin(innerImageHolder, 1)))

        for i in range(innerPoints.shape[1]):
            newOuterPoints = copy.deepcopy(outerPoints)
            innerPoint[0, 0] = innerPoints[0, i]
            innerPoint[1, 0] = innerPoints[1, i]
            #innerPoint[0, 1] = innerPoints[1, i]
            newOuterPoints = newOuterPoints - innerPoint
            distances = np.square(newOuterPoints[0,:]) + np.square(newOuterPoints[1,:])
            closestIndex = distances.index(min(distances))
            adjustedOuterImageHolder[outerPoints[0, closestIndex], outerPoints[1, closestIndex]] = 1
        innerPointCloud[fileNum, :, :] = innerImageHolder
        outerPointCloud[fileNum, :, :] = adjustedOuterImageHolder
        print('Completed ' + str(fileNum) + ' of ' + str(height))

    pointClouds = [innerPointCloud, outerPointCloud]
    wallTypes = ['Inner', 'Outer']
    for pointCloud, wallType in zip(pointClouds, wallTypes):

        with open(CSVWriteDir + PatientID + wallType + 'PointCloud' + '.csv','wb') as myfile:
            filewriter = csv.writer(myfile, delimiter=',',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
            filewriter.writerow(['X', 'Y', 'Z'])
            pointCloud = np.array(np.where(np.isin(pointCloud, 1)))
            for z, y, x in zip(pointCloud[0,:], pointCloud[1,:], pointCloud[2,:]):
                filewriter.writerow([x, y, z])
        myfile.close()
