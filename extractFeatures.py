import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import csv
import random
import math
from myFunctions import *
from scipy import ndimage, signal
from sklearn.decomposition import PCA
import datetime

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

        # Get the inner parameters if the aneurysm isn't doubled up
        if ThickInnerPointCloud[i, :, :].max() > 0 and ThickOuterPointCloud[i, :, :].max() > 0:
            if not isDoubleAAA(ThickInnerPointCloud[i, :, :]) and not isDoubleAAA(ThickOuterPointCloud[i, :, :]):
                wallVolume.append(np.where(np.isin(wallOnlyPointCloud[i, :, :], 255))[0].size)

                # Adds to inner series
                innerPerimeter.append(calcPerimeter(ThickInnerPointCloud[i, :, :]))
                innerArea.append(2 * np.sqrt(np.where(np.isin(ThickInnerPointCloud[i, :, :], 255))[0].size / math.pi))
                innerCentreLine.append(ndimage.measurements.center_of_mass(ThickInnerPointCloud[i, :, :]))

                #Adds to outer series
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
    print(AAABounds)

    # Calculates some simple length features
    featureDict['hSac'] = AAABounds['AAAEnd'] - AAABounds['AAAStart']
    featureDict['hNeck'] = AAABounds['AAAStart'] - 0
    featureDict['lSac'] = calcSegmentLength(innerCentreLine[:, AAABounds['AAAStart']: AAABounds['AAAEnd']])
    featureDict['lNeck'] = calcSegmentLength(innerCentreLine[:, 0: AAABounds['AAAStart']])
    featureDict['dOuterNeckProximal'] = outerArea[0]
    featureDict['dOuterNeckDistal'] = outerArea[AAABounds['AAAStart']]
    featureDict['dInnerNeckProximal'] = innerArea[0]
    featureDict['dInnerNeckDistal'] = innerArea[AAABounds['AAAStart']]
    featureDict['maxOuterDiameter'] = max(outerArea)
    maxDiaLocation = np.argmax(innerArea)
    print('maxDiaLocation at ' + str(maxDiaLocation))
    featureDict['bulgeHeight'] = maxDiaLocation - 0 #0 is renal artery location

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

def turnFeatureDictsToDataFrame(dictPath):
    listOfDics = [np.load(dictPath + dict).item() for dict in os.listdir(dictPath)]
    listOfKeysOfDics = [set(dict.item().keys()) for dict in listOfDics]
    uniqueKeys = set.intersection(listOfKeysOfDics)
    print('hi')


def main():
    dictPath = 'C:/Users/Luke/Documents/sharedFolder/4YP/dicts/'
    #turnFeatureDictsToDataFrame(dictPath)
    pointCloudParentDir = 'D:/processedCases/'
    renalArteryDict = {'AA':150, 'CU':150, 'DC':150, 'MH':150, 'NS':150, 'PB':150, 'PS':150, 'RR':150}
    patientListDir = sorted(os.listdir(pointCloudParentDir))
    patientListDir = [dir for dir in patientListDir if dir[0:2] in ['AA', 'CU', 'DC', 'MH', 'NS', 'PB', 'PS', 'RR']]
    for segmentedPatientDir in patientListDir:
        patientID = segmentedPatientDir[0:2]
        print('Patient ' + patientID)
        print('Renal artery at ' + str(renalArteryDict[patientID]))
        numpyPointCloudFiles = {'innerPointCloud':pointCloudParentDir + segmentedPatientDir + '/' + patientID + 'ThickInnerPointCloud.npy',
                                'outerPointCloud': pointCloudParentDir + segmentedPatientDir + '/' + patientID + 'ThickOuterPointCloud.npy'}
        myInnerPointCloud = np.load(numpyPointCloudFiles['innerPointCloud'])
        myOuterPointCloud = np.load(numpyPointCloudFiles['outerPointCloud'])
        np.save(dictPath + patientID + 'featureDict' + str(datetime.datetime.today()).split('.')[0].replace(':', '-') + '.npy', extractFeaturesFromPointClouds(myInnerPointCloud, myOuterPointCloud, renalArteryDict[patientID]))

if __name__ == "__main__":
    main()




