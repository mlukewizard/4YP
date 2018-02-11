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
    patientFeatureDict = {'PatientID':patientID}
    print('Patient ' + patientID)
    renalArteryLocation = renalArteryDict[patientID]
    print('Renal artery at ' + str(renalArteryLocation))
    numpyPointCloudFiles = ['C:\\Users\\Luke\\Documents\\sharedFolder\\4YP\\pointClouds\\Regent_' + patientID + '\\' + patientID + 'ThickInnerPointCloud.npy',
                            'C:\\Users\\Luke\\Documents\\sharedFolder\\4YP\\pointClouds\\Regent_' + patientID + '\\' + patientID + 'ThickOuterPointCloud.npy']

    ThickInnerPointCloud = np.load(numpyPointCloudFiles[0])
    ThickOuterPointCloud = np.load(numpyPointCloudFiles[1])

    # Defining the wall as the area between the outer points and the inner points
    wallOnlyPointCloud = ThickOuterPointCloud - ThickInnerPointCloud


    # Initialising arrays for the inner and outer diameters and the centre lines
    wallVolume = np.zeros(wallOnlyPointCloud.shape[0])
    innerArea = np.zeros(wallOnlyPointCloud.shape[0])
    outerArea = np.zeros(wallOnlyPointCloud.shape[0])
    outerPerimeter = np.zeros(wallOnlyPointCloud.shape[0])
    innerPerimeter = np.zeros(wallOnlyPointCloud.shape[0])
    innerCentreLine = np.zeros([2, wallOnlyPointCloud.shape[0]])
    outerCentreLine = np.zeros([2, wallOnlyPointCloud.shape[0]])

    innerCentreLinePoints = []
    possibleArcPoints = []
    foundArcCentre = False

    print('Analysing features')
    for i in range(ThickInnerPointCloud.shape[0]):
        print('Analysing slice ' + str(i))
        wallVolume[i] = np.where(np.isin(wallOnlyPointCloud[i, :, :], 255))[0].size

        labelled, num_features = ndimage.label(ThickInnerPointCloud[i, :, :], structure= np.ones([3, 3]))
        if num_features == 1 and foundArcCentre == False:
            possibleArcPoints.append([*ndimage.measurements.center_of_mass(labelled), i])
        elif num_features == 2 and foundArcCentre == False:
            foundArcCentre = True
            possibleArcPoints = np.array(possibleArcPoints)
            middleArcPoint = [int(np.average(possibleArcPoints[:, 0])), int(np.average(possibleArcPoints[:, 1])), int(np.average(possibleArcPoints[:, 2]))]
            innerCentreLinePoints.append(middleArcPoint)
            innerCentreLinePoints.insert(0, labelled*(labelled < 2))
            innerCentreLinePoints.append(labelled * (labelled > 1))
        else:
            innerCentreLinePoints.insert(0, labelled*(labelled < 2))
            if num_features == 2:
                innerCentreLinePoints.append(labelled * (labelled > 1))
            elif num_features == 3:
                print('Youve got something interesting going on here')


        # Get the inner parameters if the aneurysm isn't doubled up
        if ThickInnerPointCloud[i, :, :].max() > 0:
            if not isDoubleAAA(ThickInnerPointCloud[i, :, :]):
                innerPerimeter[i] = calcPerimeter(ThickInnerPointCloud[i, :, :])
                innerArea[i] = 2 * np.sqrt(np.where(np.isin(ThickInnerPointCloud[i, :, :], 255))[0].size / math.pi)
                innerCentreLine[:, i] = ndimage.measurements.center_of_mass(ThickInnerPointCloud[i, :, :])

        # Get the outer parameters if the aneurysm isn't doubled up
        if ThickOuterPointCloud[i, :, :].max() > 0:
            if not isDoubleAAA(ThickOuterPointCloud[i, :, :]):
                outerPerimeter[i] = calcPerimeter(ThickOuterPointCloud[i, :, :])
                outerArea[i] = 2 * np.sqrt(np.where(np.isin(ThickOuterPointCloud[i, :, :], 255))[0].size / math.pi)
                outerCentreLine[:, i] = ndimage.measurements.center_of_mass(ThickOuterPointCloud[i, :, :])

    # Simply reformatting the centre line array
    outerCentreLine = np.array([np.linspace(0, len(outerCentreLine[0])-1, len(outerCentreLine[0])), outerCentreLine[0], outerCentreLine[1]])
    innerCentreLine = np.array([np.linspace(0, len(innerCentreLine[0])-1, len(innerCentreLine[0])), innerCentreLine[0], innerCentreLine[1]])

    # Filtering the three signals because of possible inaccuracies in the function which detects if there are white blobs or one
    outerArea = signal.medfilt(outerArea, 7)
    innerArea = signal.medfilt(innerArea, 7)
    wallVolume = signal.medfilt(wallVolume, 7)

    # Automatically detects the start and end of the AAA
    AAABounds = findAAABounds(wallVolume, outerArea)
    maxDiaLocation = len(outerArea) - renalArteryLocation + np.argmax(outerArea[-renalArteryLocation:])
    print('maxDiaLocation at ' + str(maxDiaLocation))
    print(AAABounds)

    # Calculates some simple length features
    hSac = AAABounds['AAAEnd'] - AAABounds['AAAStart']
    hNeck = AAABounds['AAAStart'] - renalArteryLocation
    lSac = calcSegmentLength(innerCentreLine[:, AAABounds['AAAStart']: AAABounds['AAAEnd']])
    lNeck = calcSegmentLength(innerCentreLine[:, renalArteryLocation: AAABounds['AAAStart']])
    dOuterNeckProximal = outerArea[renalArteryLocation]
    dOuterNeckDistal = outerArea[AAABounds['AAAStart']]
    dInnerNeckProximal = innerArea[renalArteryLocation]
    dInnerNeckDistal = innerArea[AAABounds['AAAStart']]
    maxOuterDiameter = max(outerArea)
    bulgeHeight = maxDiaLocation - renalArteryLocation

    # Gets inner and outer aortic and neck volumes
    innerAorticVolume = np.where(np.isin(ThickInnerPointCloud[AAABounds['AAAStart']:AAABounds['AAAEnd'], :, :], 255))[0].size
    outerAorticVolume = np.where(np.isin(ThickOuterPointCloud[AAABounds['AAAStart']:AAABounds['AAAEnd'], :, :], 255))[0].size
    innerNeckVolume = np.where(np.isin(ThickInnerPointCloud[renalArteryLocation:AAABounds['AAAStart'], :, :], 255))[0].size
    outerNeckVolume = np.where(np.isin(ThickOuterPointCloud[renalArteryLocation:AAABounds['AAAStart'], :, :], 255))[0].size

    # Finds the tortuosity of the centre lines
    innerTortuosity = calc3DTortuosity(innerCentreLine, 9)
    outerTortuosity = calc3DTortuosity(outerCentreLine, 9)

    # Plots of the results
    plt.plot(innerArea)
    plt.show()

    plt.figure()
    ax = plt.subplot(projection='3d')
    ax.scatter(innerCentreLine[0], innerCentreLine[1], innerCentreLine[2], zdir='z', c='red', marker='.')
    plt.show()




