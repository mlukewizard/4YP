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

    #Defines the file paths
    innerBinaryDir = 'C:\\Users\\Luke\\Documents\\sharedFolder\\4YP\\Images\\Regent_'+PatientID+'\\preAugmentation\\innerBinary\\'
    CSVWriteDir = 'C:\\Users\\Luke\\Documents\\sharedFolder\\4YP\\pointClouds\\Regent_'+PatientID+'\\'
    if not os.path.exists(CSVWriteDir):
            os.mkdir(CSVWriteDir)
    outerBinaryDir = 'C:\\Users\\Luke\\Documents\\sharedFolder\\4YP\\Images\\Regent_'+PatientID+'\\preAugmentation\\outerBinary\\'

    #Extract the points for the inner aorta
    fileList = sorted(os.listdir(innerBinaryDir))
    height = len(fileList)
    sampleImage = misc.imread(innerBinaryDir + fileList[0])
    thinInnerPointCloud = np.ndarray([height, (sampleImage.shape)[0], (sampleImage.shape)[1]], dtype='uint8')
    thickInnerPointCloud = np.ndarray([height, (sampleImage.shape)[0], (sampleImage.shape)[1]], dtype='uint8')
    imageHolder = np.ndarray([(sampleImage.shape)[0], (sampleImage.shape)[1]], dtype='uint8')
    fileNum = -1

    for filename in fileList:
        fileNum = fileNum + 1
        image = misc.imread(innerBinaryDir + filename)
        thickInnerPointCloud[fileNum, :, :] = image
        imageHolder = getImagePerimeterPoints(image)
        thinInnerPointCloud[fileNum, :, :] = imageHolder
        print('Completed ' + str(fileNum) + ' of ' + str(height))

    #extract the points for the outer aorta
    fileList = sorted(os.listdir(outerBinaryDir))
    height = len(fileList)
    sampleImage = misc.imread(outerBinaryDir + fileList[0])
    thinOuterPointCloud = np.ndarray([height, (sampleImage.shape)[0], (sampleImage.shape)[1]], dtype='uint8')
    thickOuterPointCloud = np.ndarray([height, (sampleImage.shape)[0], (sampleImage.shape)[1]], dtype='uint8')
    imageHolder = np.ndarray([(sampleImage.shape)[0], (sampleImage.shape)[1]], dtype='uint8')
    fileNum = -1

    for filename in fileList:
        fileNum = fileNum + 1
        image = misc.imread(outerBinaryDir + filename)
        thickOuterPointCloud[fileNum, :, :] = image
        imageHolder = getImagePerimeterPoints(image)
        thinOuterPointCloud[fileNum, :, :] = imageHolder
        print('Completed ' + str(fileNum) + ' of ' + str(height))

    pointClouds = [thinInnerPointCloud, thinOuterPointCloud, thickInnerPointCloud, thickOuterPointCloud]
    wallTypes = ['ThinInner', 'ThinOuter', 'ThickInner', 'ThickOuter']
    for pointCloud, wallType in zip(pointClouds, wallTypes):
        np.save(CSVWriteDir + PatientID + wallType + 'PointCloud', pointCloud)


    pointClouds = [thinInnerPointCloud, thinOuterPointCloud]
    wallTypes = ['ThinInner', 'ThinOuter']
    for pointCloud, wallType in zip(pointClouds, wallTypes):
        with open(CSVWriteDir + PatientID + wallType + 'PointCloud' + '.csv', 'wb') as myfile:
           myfile = open(CSVWriteDir + PatientID + wallType + 'PointCloud' + '.csv', 'wb')
           filewriter = csv.writer(myfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
           filewriter.writerow(['X', 'Y', 'Z'])
           pointCloud = np.array(np.where(np.isin(pointCloud, 255)))
           for z, y, x in zip(pointCloud[0,:], pointCloud[1, :], pointCloud[2, :]):
              filewriter.writerow([x, y, z])
        myfile.close()

    '''
    #Write innerPointCloud
    with open(CSVWriteDir + PatientID + 'Inner' + 'PointCloud' + '.csv','wb') as myfile:
        filewriter = csv.writer(myfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        filewriter.writerow(['X', 'Y', 'Z'])
        for i in range(innerPointCloud.shape[0]):
            for j in range(innerPointCloud.shape[1]):
                for k in range(innerPointCloud.shape[2]):
                    if innerPointCloud[i,j,k] > 0:
                        filewriter.writerow([j, k, i])
    myfile.close()

    #Write outerPointCloud
    with open(CSVWriteDir + PatientID + 'Outer' + 'PointCloud' + '.csv','wb') as myfile:
        filewriter = csv.writer(myfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        filewriter.writerow(['X', 'Y', 'Z'])
        for i in range(outerPointCloud.shape[0]):
            for j in range(outerPointCloud.shape[1]):
                for k in range(outerPointCloud.shape[2]):
                    if outerPointCloud[i,j,k] > 0:
                       filewriter.writerow([j, k, i])
    myfile.close()

    '''
    #plt.figure()
    #z,x,y = innerPointCloud.nonzero()
    #ax = plt.subplot(111, projection='3d')
    #ax.scatter(x, y, -z, zdir = 'z', c = 'red')
    #plt.show()