import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import csv
import random
import math
from myFunctions import *
from scipy import ndimage, signal
from sklearn.decomposition import PCA

def extractFeaturesFromPointClouds(ThickInnerPointCloud, ThickOuterPointCloud, renalArteryLocation):
    featureDict = {}
    ThickInnerPointCloud = ThickInnerPointCloud[renalArteryLocation:]
    ThickOuterPointCloud = ThickOuterPointCloud[renalArteryLocation:]

    # Defining the wall as the area between the outer points and the inner points
    wallOnlyPointCloud = ThickOuterPointCloud - ThickInnerPointCloud

    # Initialising arrays for the inner and outer diameters and the centre lines
    wallVolume = []
    innerArea = []
    outerArea = []
    outerPerimeter = []
    innerPerimeter = []
    innerCentreLine = []
    outerCentreLine = []

    print('Analysing features')
    for i in range(ThickInnerPointCloud.shape[0]):
        print('Analysing slice ' + str(i))
        wallVolume.append(np.where(np.isin(wallOnlyPointCloud[i, :, :], 255))[0].size)

        # Get the inner parameters if the aneurysm isn't doubled up
        if ThickInnerPointCloud[i, :, :].max() > 0:
            if not isDoubleAAA(ThickInnerPointCloud[i, :, :]):
                innerPerimeter.append(calcPerimeter(ThickInnerPointCloud[i, :, :]))
                innerArea.append(2 * np.sqrt(np.where(np.isin(ThickInnerPointCloud[i, :, :], 255))[0].size / math.pi))
                innerCentreLine.append(ndimage.measurements.center_of_mass(ThickInnerPointCloud[i, :, :]))

        # Get the outer parameters if the aneurysm isn't doubled up
        if ThickOuterPointCloud[i, :, :].max() > 0:
            if not isDoubleAAA(ThickOuterPointCloud[i, :, :]):
                outerPerimeter.append(calcPerimeter(ThickOuterPointCloud[i, :, :]))
                outerArea.append(2 * np.sqrt(np.where(np.isin(ThickOuterPointCloud[i, :, :], 255))[0].size / math.pi))
                outerCentreLine.append(ndimage.measurements.center_of_mass(ThickOuterPointCloud[i, :, :]))

    # Turning centreline from list to numpy array with extra dimension which is z axis
    innerCentreLine = np.concatenate([np.expand_dims(np.arange(len(innerCentreLine)), axis=1), np.array(innerCentreLine)], axis=1)
    outerCentreLine = np.concatenate([np.expand_dims(np.arange(len(outerCentreLine)), axis=1), np.array(outerCentreLine)], axis=1)

    #Turning all other series in numpy arrays
    wallVolume = np.array(wallVolume)
    innerArea = np.array(innerArea)
    outerArea = np.array(outerArea)
    outerPerimeter = np.array(outerPerimeter)
    innerPerimeter = np.array(innerPerimeter)
    innerCentreLine = np.array(innerCentreLine)
    outerCentreLine = np.array(outerCentreLine)

    # Automatically detects the start and end of the AAA
    AAABounds = findAAABounds(wallVolume, outerArea)
    maxDiaLocation = np.argmax(outerArea)
    print('maxDiaLocation at ' + str(maxDiaLocation))
    print(AAABounds)

    # Calculates some simple length features
    featureDict['hSac'] = AAABounds['AAAEnd'] - AAABounds['AAAStart']
    featureDict['hNeck'] = AAABounds['AAAStart'] - renalArteryLocation
    featureDict['lSac'] = calcSegmentLength(innerCentreLine[:, AAABounds['AAAStart']: AAABounds['AAAEnd']])
    featureDict['lNeck'] = calcSegmentLength(innerCentreLine[:, renalArteryLocation: AAABounds['AAAStart']])
    featureDict['dOuterNeckProximal'] = outerArea[renalArteryLocation]
    featureDict['dOuterNeckDistal'] = outerArea[AAABounds['AAAStart']]
    featureDict['dInnerNeckProximal'] = innerArea[renalArteryLocation]
    featureDict['dInnerNeckDistal'] = innerArea[AAABounds['AAAStart']]
    featureDict['maxOuterDiameter'] = max(outerArea)
    featureDict['bulgeHeight'] = maxDiaLocation - renalArteryLocation

    # Gets inner and outer aortic and neck volumes
    featureDict['innerAorticVolume'] = np.where(np.isin(ThickInnerPointCloud[AAABounds['AAAStart']:AAABounds['AAAEnd'], :, :], 255))[0].size
    featureDict['outerAorticVolume'] = np.where(np.isin(ThickOuterPointCloud[AAABounds['AAAStart']:AAABounds['AAAEnd'], :, :], 255))[0].size
    featureDict['AAAThrombusVolume'] = np.where(np.isin(wallOnlyPointCloud[AAABounds['AAAStart']:AAABounds['AAAEnd'], :, :], 255))[0].size
    featureDict['innerNeckVolume'] = np.where(np.isin(ThickInnerPointCloud[0:AAABounds['AAAStart'], :, :], 255))[0].size
    featureDict['outerNeckVolume'] = np.where(np.isin(ThickOuterPointCloud[0:AAABounds['AAAStart'], :, :], 255))[0].size
    featureDict['NeckThrombusVolume'] = np.where(np.isin(wallOnlyPointCloud[0:AAABounds['AAAStart'], :, :], 255))[0].size

    # Finds the tortuosity of the centre lines
    featureDict['innerTortuosity'] = calc3DTortuosity(innerCentreLine, 9)
    featureDict['outerTortuosity'] = calc3DTortuosity(outerCentreLine, 9)

    # Plots of the results
    plt.plot(innerArea)
    plt.show()

    plt.figure()
    ax = plt.subplot(projection='3d')
    ax.scatter(innerCentreLine[:, 0], innerCentreLine[:, 1], innerCentreLine[:, 2], zdir='z', c='red', marker='.')
    plt.show()
    return featureDict

def main():
    patientList = ['RR', 'PS', 'PB', 'NS', 'DC']
    renalArteryDict = {'PS': 150, 'PB': 151, 'NS': 152, 'DC': 153, 'RR': 154}
    patientDict = {}
    for patientID in patientList:
        print('Patient ' + patientID)
        print('Renal artery at ' + str(renalArteryDict[patientID]))
        numpyPointCloudFiles = ['C:\\Users\\Luke\\Documents\\sharedFolder\\4YP\\pointClouds\\Regent_' + patientID + '\\' + patientID + 'ThickInnerPointCloud.npy',
                                'C:\\Users\\Luke\\Documents\\sharedFolder\\4YP\\pointClouds\\Regent_' + patientID + '\\' + patientID + 'ThickOuterPointCloud.npy']

        myInnerPointCloud = np.load(numpyPointCloudFiles[0])
        myOuterPointCloud = np.load(numpyPointCloudFiles[1])
        patientDict[patientID] = extractFeaturesFromPointClouds(myInnerPointCloud, myOuterPointCloud, renalArteryDict[patientID])


if __name__ == "__main__":
    main()




