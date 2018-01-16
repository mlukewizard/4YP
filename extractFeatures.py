import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import csv
import random
import math
from myFunctions import *
from scipy import ndimage, signal


patientList = ['RR', 'PS', 'PB', 'NS', 'DC']
renalArteryDict = {'PS': 150, 'PB': 151, 'NS': 152, 'DC': 153, 'RR': 154}

for patientID in patientList:
    print('Patient ' + patientID)
    renalArteryLocation = renalArteryDict[patientID]
    print('Renal artery at ' + str(renalArteryLocation))
    numpyPointCloudFiles = ['C:\\Users\\Luke\\Documents\\sharedFolder\\4YP\\pointClouds\\Regent_' + patientID + '\\' + patientID + 'ThickInnerPointCloud.npy',
                            'C:\\Users\\Luke\\Documents\\sharedFolder\\4YP\\pointClouds\\Regent_' + patientID + '\\' + patientID + 'ThickOuterPointCloud.npy',
                            'C:\\Users\\Luke\\Documents\\sharedFolder\\4YP\\pointClouds\\Regent_' + patientID + '\\' + patientID + 'ThinInnerPointCloud.npy',
                            'C:\\Users\\Luke\\Documents\\sharedFolder\\4YP\\pointClouds\\Regent_' + patientID + '\\' + patientID + 'ThinOuterPointCloud.npy']

    '''
    csvPointCloudFiles = ['C:\\Users\\Luke\\Documents\\sharedFolder\\4YP\\pointClouds\\Regent_' + patientID + '\\' + patientID + 'ThinInnerPointCloud.csv',
                          'C:\\Users\\Luke\\Documents\\sharedFolder\\4YP\\pointClouds\\Regent_' + patientID + '\\' + patientID + 'ThinInnerPointCloud.csv']
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
    '''

    ThickInnerPointCloud = np.load(numpyPointCloudFiles[0])
    ThickOuterPointCloud = np.load(numpyPointCloudFiles[1])
    ThinInnerPointCloud = np.load(numpyPointCloudFiles[2])
    ThinOuterPointCloud = np.load(numpyPointCloudFiles[3])

    # Defining the wall as the area between the outer points and the inner points
    wallOnlyPointCloud = ThickOuterPointCloud - ThickInnerPointCloud

    # Initialising arrays for the inner and outer diameters and the centre lines
    wallVolume = np.zeros(wallOnlyPointCloud.shape[0])
    avgInnerDiameter = np.zeros(wallOnlyPointCloud.shape[0])
    avgOuterDiameter = np.zeros(wallOnlyPointCloud.shape[0])
    outerPerimeter = np.zeros(wallOnlyPointCloud.shape[0])
    innerPerimeter = np.zeros(wallOnlyPointCloud.shape[0])
    innerCentreLine = np.zeros([2, wallOnlyPointCloud.shape[0]])
    outerCentreLine = np.zeros([2, wallOnlyPointCloud.shape[0]])
    print('Analysing features')
    for i in range(ThickInnerPointCloud.shape[0]):
        print('Analysing slice ' + str(i))
        wallVolume[i] = np.where(np.isin(wallOnlyPointCloud[i, :, :], 255))[0].size

        # Get the inner parameters if the aneurysm isn't doubled up
        if ThickInnerPointCloud[i, :, :].max() > 0:
            # Just checking out how dense the points are in the thin point cloud
            if not isDoubleAAA(ThickInnerPointCloud[i, :, :]):
                #innerPerimeter[i] = calcPerimeter(ThickInnerPointCloud[i, :, :])
                avgInnerDiameter[i] = 2 * np.sqrt(np.where(np.isin(ThickInnerPointCloud[i, :, :], 255))[0].size / math.pi)
                innerCentreLine[:, i] = ndimage.measurements.center_of_mass(ThickInnerPointCloud[i, :, :])
                print('innerPerimeter is ' + str(innerPerimeter[i]) + ' and innerDiameter is ' + str(avgInnerDiameter[i]))

        # Get the outer parameters if the aneurysm isn't doubled up
        if ThickOuterPointCloud[i, :, :].max() > 0:
            if not isDoubleAAA(ThickOuterPointCloud[i, :, :]):
                #outerPerimeter[i] = calcPerimeter(ThickOuterPointCloud[i, :, :])
                avgOuterDiameter[i] = 2 * np.sqrt(np.where(np.isin(ThickOuterPointCloud[i, :, :], 255))[0].size / math.pi)
                outerCentreLine[:, i] = ndimage.measurements.center_of_mass(ThickOuterPointCloud[i, :, :])

    # Simply reformatting the centre line array
    outerCentreLine = np.array([np.linspace(0, len(outerCentreLine[0])-1, len(outerCentreLine[0])), outerCentreLine[0], outerCentreLine[1]])
    innerCentreLine = np.array([np.linspace(0, len(innerCentreLine[0])-1, len(innerCentreLine[0])), innerCentreLine[0], innerCentreLine[1]])

    # Filtering the three signals because of possible inaccuracies in the function which detects if there are white blobs or one
    avgOuterDiameter = signal.medfilt(avgOuterDiameter, 7)
    avgInnerDiameter = signal.medfilt(avgInnerDiameter, 7)
    wallVolume = signal.medfilt(wallVolume, 7)

    # Automatically detects the start and end of the AAA
    AAABounds = findAAABounds(wallVolume, avgOuterDiameter)
    maxDiaLocation = len(avgOuterDiameter) - renalArteryLocation + np.argmax(avgOuterDiameter[-renalArteryLocation:])
    print('maxDiaLocation at ' + str(maxDiaLocation))
    print(AAABounds)

    # Calculates some simple length features
    hSac = AAABounds[1] - AAABounds[0]
    hNeck = AAABounds[0] - renalArteryLocation
    lSac = calcSegmentLength(innerCentreLine[:, AAABounds[0]: AAABounds[1]])
    lNeck = calcSegmentLength(innerCentreLine[:, renalArteryLocation: AAABounds[0]])
    dOuterNeckProximal = avgOuterDiameter[renalArteryLocation]
    dOuterNeckDistal = avgOuterDiameter[AAABounds[0]]
    dInnerNeckProximal = avgInnerDiameter[renalArteryLocation]
    dInnerNeckDistal = avgInnerDiameter[AAABounds[0]]
    maxOuterDiameter = max(avgOuterDiameter)
    bulgeHeight = maxDiaLocation - renalArteryLocation

    # Gets inner and outer aortic and neck volumes
    innerAorticVolume = np.where(np.isin(ThickInnerPointCloud[AAABounds[0]:AAABounds[1], :, :], 255))[0].size
    outerAorticVolume = np.where(np.isin(ThickOuterPointCloud[AAABounds[0]:AAABounds[1], :, :], 255))[0].size
    innerNeckVolume = np.where(np.isin(ThickInnerPointCloud[renalArteryLocation:AAABounds[0], :, :], 255))[0].size
    outerNeckVolume = np.where(np.isin(ThickOuterPointCloud[renalArteryLocation:AAABounds[0], :, :], 255))[0].size

    # Finds the tortuosity of the centre lines
    innerTortuosity = calc3DTortuosity(innerCentreLine, 9)
    outerTortuosity = calc3DTortuosity(outerCentreLine, 9)

    # Plots of the results
    plt.plot(avgInnerDiameter)
    plt.show()

    plt.figure()
    ax = plt.subplot(projection='3d')
    ax.scatter(innerCentreLine[0], innerCentreLine[1], innerCentreLine[2], zdir='z', c='red', marker='.')
    plt.show()




