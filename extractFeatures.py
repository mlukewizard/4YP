import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import csv
import random
import math
from myFunctions import *
from scipy import ndimage

patientID = 'RR'
csvPointCloudFiles = ['C:\\Users\\Luke\\Documents\\sharedFolder\\4YP\\pointClouds\\Regent_' + patientID + '\\' + patientID + 'ThinInnerPointCloud.csv',
                      'C:\\Users\\Luke\\Documents\\sharedFolder\\4YP\\pointClouds\\Regent_' + patientID + '\\' + patientID + 'ThinInnerPointCloud.csv']
numpyPointCloudFiles = ['C:\\Users\\Luke\\Documents\\sharedFolder\\4YP\\pointClouds\\Regent_' + patientID + '\\' + patientID + 'ThickInnerPointCloud.npy',
                        'C:\\Users\\Luke\\Documents\\sharedFolder\\4YP\\pointClouds\\Regent_' + patientID + '\\' + patientID + 'ThickOuterPointCloud.npy']

pointList = []
pointClouds = []

for pointCloudFile in csvPointCloudFiles:
    with open(pointCloudFile, 'rb') as csvfile:
        fileReader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        next(fileReader)
        for row in fileReader:
            if random.uniform(0, 1) > 0.9:
                row = row[0].split(',')
                row = [int(row[0]), int(row[1]), int(row[2])]
                pointList.append(row)
        pointClouds.append(np.array(pointList))

ThickInnerPointCloud = np.load(numpyPointCloudFiles[0])
ThickOuterPointCloud = np.load(numpyPointCloudFiles[1])

wallOnlyPointCloud = ThickOuterPointCloud - ThickInnerPointCloud
wallVolume = np.zeros(wallOnlyPointCloud.shape[0])
avgInnerDiameter = np.zeros(wallOnlyPointCloud.shape[0])
avgOuterDiameter = np.zeros(wallOnlyPointCloud.shape[0])
innerCentreLine = np.zeros([2, wallOnlyPointCloud.shape[0]])
outerCentreLine = np.zeros([2, wallOnlyPointCloud.shape[0]])
for i in range(wallOnlyPointCloud.shape[0]):
    print('Analysing slice ' + str(i))
    wallVolume[i] = np.where(np.isin(wallOnlyPointCloud[i, :, :], 255))[0].size

    #Get the inner parameters if the aneurysm isnt doubled up
    if ThickInnerPointCloud[i, :, :].max() > 0:
        if not isDoubleAAA(ThickInnerPointCloud[i, :, :]):
            avgInnerDiameter[i] = 2 * np.sqrt(np.where(np.isin(ThickInnerPointCloud[i, :, :], 255))[0].size / math.pi)
            innerCentreLine[:, i] = ndimage.measurements.center_of_mass(ThickInnerPointCloud[i, :, :])

    #Get the outer parameters if the aneurysm isnt doubled up
    if ThickOuterPointCloud[i, :, :].max() > 0:
        if not isDoubleAAA(ThickOuterPointCloud[i, :, :]):
            avgOuterDiameter[i] = 2 * np.sqrt(np.where(np.isin(ThickOuterPointCloud[i, :, :], 255))[0].size / math.pi)
            outerCentreLine[:, i] = ndimage.measurements.center_of_mass(ThickOuterPointCloud[i, :, :])
        #slice = ThickInnerPointCloud[i, :, :]
        #plt.imshow(slice, cmap='gray')
        #plt.show()



#plt.plot(wallVolume)
plt.plot(avgInnerDiameter)
plt.plot(avgOuterDiameter)
plt.show()



#plt.figure()
#ax = plt.subplot(projection='3d')
#ax.scatter(pointClouds[1][:, 0], pointClouds[1][:, 1], -pointClouds[1][:, 2], zdir='z', c='red', marker='.')
#plt.show()

