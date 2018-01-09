import os, shutil
import csv
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import ndimage
from imageProcessingFuntions import getFolderCoM, getImagePerimeterPoints
import scipy.misc as misc
from PIL import Image, ImageEnhance, ImageFilter
from plyfile import PlyData, PlyElement
from mpl_toolkits.mplot3d import Axes3D

patientList = ['NS', 'DC', 'PB', 'PS', 'RR']
for PatientID in patientList:
    print('Extracting patient ' + PatientID)
    innerBinaryDir = '/media/sf_sharedFolder/4YP/Images/Regent_'+PatientID+'/preAugmentation/innerBinary/'
    CSVWriteDir = '/media/sf_sharedFolder/4YP/pointClouds/Regent_'+PatientID+'/'
    if not os.path.exists(CSVWriteDir):
            os.mkdir(CSVWriteDir)
    outerBinaryDir = '/media/sf_sharedFolder/4YP/Images/Regent_'+PatientID+'/preAugmentation/outerBinary/'

    #Extract the points for the inner aorta
    fileList = sorted(os.listdir(innerBinaryDir))
    height = len(fileList)
    sampleImage = misc.imread(innerBinaryDir + fileList[0])
    innerPointCloud = np.ndarray([height, (sampleImage.shape)[0], (sampleImage.shape)[1]])
    imageHolder = np.ndarray([(sampleImage.shape)[0], (sampleImage.shape)[1]])
    fileNum = -1

    for filename in fileList:
        fileNum = fileNum + 1
        image = misc.imread(innerBinaryDir + filename)
        imageHolder = getImagePerimeterPoints(image)

        innerPointCloud[fileNum, :, :] = imageHolder
        print('Completed ' + str(fileNum) + ' of ' + str(height))

    #extract the points for the outer aorta
    fileList = sorted(os.listdir(outerBinaryDir))
    height = len(fileList)
    sampleImage = misc.imread(outerBinaryDir + fileList[0])
    outerPointCloud = np.ndarray([height, (sampleImage.shape)[0], (sampleImage.shape)[1]])
    imageHolder = np.ndarray([(sampleImage.shape)[0], (sampleImage.shape)[1]])
    fileNum = -1

    for filename in fileList:
        fileNum = fileNum + 1
        image = misc.imread(outerBinaryDir + filename)
        imageHolder = getImagePerimeterPoints(image)

        outerPointCloud[fileNum, :, :] = imageHolder
        print('Completed ' + str(fileNum) + ' of ' + str(height))

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


    #plt.figure()
    #z,x,y = innerPointCloud.nonzero()
    #ax = plt.subplot(111, projection='3d')
    #ax.scatter(x, y, -z, zdir = 'z', c = 'red')
    #plt.show()
