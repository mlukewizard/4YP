import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import csv
import copy
import random
import os
import math
from scipy import ndimage
#from myFunctions import *
from scipy import ndimage, signal
from sklearn.decomposition import PCA
import datetime
from PIL import Image, ImageFilter

def extractFeaturesFromPointClouds(ThickInnerPointCloud, ThickOuterPointCloud, renalArteryLocation, sliceThickness, pixelSpacing):
    featureDict = {}

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
        if ThickInnerPointCloud[i, :, :].max() > 0 and ThickOuterPointCloud[i, :, :].max() > 0 and not isDoubleAAA(ThickInnerPointCloud[i, :, :]) and not isDoubleAAA(ThickOuterPointCloud[i, :, :]):
            wallVolume.append(np.where(np.isin(wallOnlyPointCloud[i, :, :], 255))[0].size)

            # Adds to inner series
            innerPerimeter.append(calcPerimeter(ThickInnerPointCloud[i, :, :]))
            innerArea.append(2 * np.sqrt(np.where(np.isin(ThickInnerPointCloud[i, :, :], 255))[0].size / math.pi))
            innerCentreLine.append(ndimage.measurements.center_of_mass(ThickInnerPointCloud[i, :, :]))

            #Adds to outer series
            outerPerimeter.append(calcPerimeter(ThickOuterPointCloud[i, :, :]))
            outerArea.append(2 * np.sqrt(np.where(np.isin(ThickOuterPointCloud[i, :, :], 255))[0].size / math.pi))
            outerCentreLine.append(ndimage.measurements.center_of_mass(ThickOuterPointCloud[i, :, :]))

        else:
            wallVolume.append(float('nan'))
            # Adds NaN to inner series
            innerPerimeter.append(float('nan'))
            innerArea.append(float('nan'))
            innerCentreLine.append((float('nan'), float('nan')))
            # Adds NaN to outer series
            outerPerimeter.append(float('nan'))
            outerArea.append(float('nan'))
            outerCentreLine.append((float('nan'), float('nan')))

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
    AAABounds = findAAABounds(wallVolume, outerArea, renalArteryLocation)
    print(AAABounds)

    # Calculates some simple length features
    featureDict['hSac'] = AAABounds['AAAEnd'] - AAABounds['AAAStart'] * sliceThickness
    featureDict['hNeck'] = AAABounds['AAAStart'] - renalArteryLocation
    featureDict['lSac'] = calcSegmentLength(innerCentreLine[:, AAABounds['AAAStart']: AAABounds['AAAEnd']])
    featureDict['lNeck'] = calcSegmentLength(innerCentreLine[:, renalArteryLocation: AAABounds['AAAStart']])
    featureDict['dOuterNeckProximal'] = outerArea[renalArteryLocation]
    featureDict['dOuterNeckDistal'] = outerArea[AAABounds['AAAStart']]
    featureDict['dInnerNeckProximal'] = innerArea[renalArteryLocation]
    featureDict['dInnerNeckDistal'] = innerArea[AAABounds['AAAStart']]
    featureDict['maxOuterArea'] = np.nanmax(outerArea) * pixelSpacing[0] * pixelSpacing[1]
    featureDict['maxInnerArea'] = np.nanmax(outerArea) * pixelSpacing[0] * pixelSpacing[1]
    maxDiaLocation = innerArea[0:renalArteryLocation].size + np.nanargmax(innerArea[renalArteryLocation:])
    print('maxDiaLocation at ' + str(maxDiaLocation))
    featureDict['bulgeHeight'] = (maxDiaLocation - renalArteryLocation) * sliceThickness

    # Gets inner and outer aortic and neck volumes
    featureDict['innerAorticVolume'] = np.where(np.isin(ThickInnerPointCloud[AAABounds['AAAStart']:AAABounds['AAAEnd'], :, :], 255))[0].size
    featureDict['outerAorticVolume'] = np.where(np.isin(ThickOuterPointCloud[AAABounds['AAAStart']:AAABounds['AAAEnd'], :, :], 255))[0].size
    featureDict['AAAThrombusVolume'] = np.where(np.isin(wallOnlyPointCloud[AAABounds['AAAStart']:AAABounds['AAAEnd'], :, :], 255))[0].size
    featureDict['innerNeckVolume'] = np.where(np.isin(ThickInnerPointCloud[renalArteryLocation:AAABounds['AAAStart'], :, :], 255))[0].size
    featureDict['outerNeckVolume'] = np.where(np.isin(ThickOuterPointCloud[renalArteryLocation:AAABounds['AAAStart'], :, :], 255))[0].size
    featureDict['NeckThrombusVolume'] = np.where(np.isin(wallOnlyPointCloud[renalArteryLocation:AAABounds['AAAStart'], :, :], 255))[0].size

    # Finds the tortuosity of the centre lines
    featureDict['AAAInnerTortuosity'] = calc3DTortuosity(innerCentreLine[AAABounds['AAAStart']:AAABounds['AAAEnd']], 9)
    featureDict['AAAOuterTortuosity'] = calc3DTortuosity(outerCentreLine[AAABounds['AAAStart']:AAABounds['AAAEnd']], 9)

    '''
    # Plots of the results
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.gca().set_title('innerArea')
    plt.plot(innerArea)
    plt.subplot(1, 3, 2)
    plt.gca().set_title('outerArea')
    plt.plot(outerArea)
    plt.subplot(1, 3, 3)
    plt.gca().set_title('wallVolume')
    plt.plot(wallVolume)
    plt.show()

    plt.figure()
    ax = plt.subplot(projection='3d')
    ax.scatter(sliceThickness*innerCentreLine[:, 0], innerCentreLine[:, 1], innerCentreLine[:, 2], zdir='z', c='red', marker='.')
    plt.show()
    '''

    return featureDict

def lukesShell(image):
    shelledImage = image[0:-2, 0:-2] + image[2:, 2:] + image[0:-2, 2:] + image[2:, 0:-2]
    shelledImage = np.pad(shelledImage, 1, 'edge')
    shelledImage = np.where(shelledImage > 0, 255, 0) - np.where(image > 0, 255, 0)

def calcPerimeter(image):
    image = shell(image)
    whitePoints = np.array(np.where(np.isin(image, 255)))
    perimeter = 0
    continueSumming = True
    currentPoint = np.zeros([2, 1], dtype='int')
    currentPoint[0] = np.where(np.isin(image, 255))[0][0]
    currentPoint[1] = np.where(np.isin(image, 255))[1][0]
    while continueSumming:
        whitePoints = whitePoints - currentPoint
        distances = np.sqrt(np.square(whitePoints[0, :]) + np.square(whitePoints[1, :]))
        closestIndex = np.argmin(distances)
        perimeter = perimeter + min(distances)
        currentPoint[0] = whitePoints[0][closestIndex]
        currentPoint[1] = whitePoints[1][closestIndex]
        whitePoints = np.delete(whitePoints, closestIndex, 1)
        if len(whitePoints[0]) == 0:
            continueSumming = False
    return perimeter

def getImagePerimeterPoints(inputImage):
    image = Image.fromarray(inputImage).convert('L')
    image = image.filter(ImageFilter.FIND_EDGES)
    outputImage = np.array(image)
    return outputImage

def shell(image):
    image2 = getImagePerimeterPoints(image)
    xPoints = np.where(np.isin(image, 255))[0]
    yPoints = np.where(np.isin(image, 255))[1]
    for i, j in zip(xPoints, yPoints):
        conditionList = [image2[i, j-1] != 0, image2[i+1, j] != 0, image2[i, j+1] != 0, image2[i-1, j] != 0]
        if sum(conditionList) > 1:
            image2[i, j] = 0
    return image2

def calcSegmentLength(line):
    length = 0
    for j in range(line.shape[1]-1):
        length = length + np.sqrt(np.square(line[0, j+1] - line[0, j]) +
                        np.square(line[1, j+1] - line[1, j]) +
                        np.square(line[2, j+1] - line[2, j]))
    return length

def calc3DTortuosity(centreLine, windowLength):
    centreLineTortuosity = np.ndarray([centreLine.shape[1]])
    halfWindow = int(math.floor(windowLength/2))
    paddedCentreLine = np.pad(centreLine, halfWindow, 'edge')[halfWindow:-halfWindow, :]
    for i in range(centreLine.shape[1]):
        evalPoint = i + halfWindow
        absoluteDistance = np.sqrt(np.square(paddedCentreLine[0, evalPoint+halfWindow] - paddedCentreLine[0, evalPoint-halfWindow]) +
            np.square(paddedCentreLine[1, evalPoint + halfWindow] - paddedCentreLine[1, evalPoint - halfWindow]) +
            np.square(paddedCentreLine[2, evalPoint + halfWindow] - paddedCentreLine[2, evalPoint - halfWindow]))
        curveDistance = 0
        for j in range(windowLength-1):
            curveDistance = curveDistance + np.sqrt(np.square(paddedCentreLine[0, i+j+1] - paddedCentreLine[0, i+j]) +
                                                    np.square(paddedCentreLine[1, i+j+1] - paddedCentreLine[1, i+j]) +
                                                    np.square(paddedCentreLine[2, i+j+1] - paddedCentreLine[2, i+j]))
        tortuosity = curveDistance / absoluteDistance
        centreLineTortuosity[i] = tortuosity
    return centreLineTortuosity

def findAAABounds(wallVolume, OuterArea, renalArteryLocation):
    maxDia = len(OuterArea[0:renalArteryLocation]) + np.nanargmax(OuterArea[renalArteryLocation:])
    maxVol = len(wallVolume[0:renalArteryLocation]) + np.nanargmax(wallVolume[renalArteryLocation:])
    if maxDia == maxVol:
        print('This is probably incorrect')
    acceptableSteps = np.linspace(1, 50, 50, dtype='int').tolist()
    timeSerieses = [wallVolume, OuterArea, wallVolume, OuterArea]
    directions = [-1, -1, 1, 1]
    points = []
    maxLocations = [maxVol, maxDia, maxVol, maxDia]
    for timeSeries, direction, start in zip(timeSerieses, directions, maxLocations):
        looking = True
        copyStart = copy.deepcopy(start)
        while looking:
            if any(((np.append(timeSeries, np.zeros(60, dtype = 'int'))[copyStart + direction*step] - np.append(timeSeries, np.zeros(60, dtype = 'int'))[copyStart])/step < -np.nanmax(timeSeries)/200 and np.append(timeSeries, np.zeros(60, dtype = 'int'))[copyStart + direction*step] != 0) for step in acceptableSteps):
                copyStart = copyStart + direction
            else:
                looking = False
        points.append(copyStart)
    if abs(points[3] - points[2]) > 10:
        print('Warning: Your algorithm is unsure about where the aneurysm ends, wall thickness suggests ' + str(points[2]) + ' and area suggests ' + str(points[3]))
    if abs(points[1] - points[0]) > 10:
        print('Warning: Your algorithm is unsure about where the aneurysm starts, wall thickness suggests ' + str(points[0]) + ' and area suggests ' + str(points[1]))
    return {'AAAStart':int((points[0] + points[1])/2), 'AAAEnd':int((points[2] + points[3])/2)}
    #return [int((points[0] + points[1])/2), int((points[2] + points[3])/2)]

def isDoubleAAA(image):
    from scipy import ndimage
    if np.max(image) != 255:
        return False
    newPC, num_features = ndimage.label(image)
    if num_features > 1:
        if len(np.where(newPC == 1)[0]) > 10 and len(np.where(newPC == 2)[0]) > 10:
            return True
    else:
        return False


def main():
    dictPath = 'C:/Users/Luke/Documents/sharedFolder/4YP/dicts/'
    spacingDictDir = 'C:/Users/Luke/Documents/sharedFolder/4YP/Spacings.npy'
    thicknessDictDir = 'C:/Users/Luke/Documents/sharedFolder/4YP/Thicknesses.npy'
    pointCloudParentDir = 'D:/processedCases/'
    renalArteryDict = {'DC': 162, 'NS ': 218, 'PB': 229, 'PS': 226, 'RR': 194, 'GC': 256, 'MH': 226, 'AA': 196, 'AD': 213, 'AE': 238, 'AF': 238, 'AG': 204, 'AI': 276,
                       'AJ': 205, 'CC': 100, 'CE': 263, 'CG': 259, 'CI': 276, 'CK': 208, 'CM': 272, 'CO': 266, 'CQ': 245, 'CS': 200, 'CU': 266, 'CW': 237, 'XC': 239,
                       'DK': 282, 'DM': 250, 'DQ': 252, 'DW': 261, 'EI': 272, 'EK': 234, 'EO': 274, 'EQ': 259, 'ES': 290, 'EU': 234, 'EY': 348, 'FA': 251, 'FG': 86,
                       'FI': 261, 'FO': 237, 'FQ': 238, 'FS': 252, 'FW': 238, 'GA': 240, 'YC': 221, 'GQ': 85, 'GY': 118, 'HA': 275, 'HC': 274, 'HG': 223, 'HI': 289,
                       'HS': 241, }
    pixelSpacingDict = np.load(spacingDictDir).item()
    thicknessDict = np.load(thicknessDictDir).item()
    patientListDir = sorted(os.listdir(pointCloudParentDir))
    patientListDir = [dir for dir in patientListDir if dir[0:2] in renalArteryDict.keys()]
    #patientListDir = [dir for dir in patientListDir if dir[0:2] in ['AE']]
    print('Weve got ' + str(len(renalArteryDict.keys())) + ' patients')
    for segmentedPatientDir in patientListDir:
        patientID = segmentedPatientDir[0:2]
        print('Patient ' + patientID)
        print('Renal artery at ' + str(renalArteryDict[patientID]))
        numpyPointCloudFiles = {'innerPointCloud':pointCloudParentDir + segmentedPatientDir + '/' + patientID + 'ThickInnerPointCloud.npy',
                                'outerPointCloud': pointCloudParentDir + segmentedPatientDir + '/' + patientID + 'ThickOuterPointCloud.npy'}
        myInnerPointCloud = np.load(numpyPointCloudFiles['innerPointCloud'])
        myInnerPointCloud = np.where(myInnerPointCloud > 0, 255, 0)
        myOuterPointCloud = np.load(numpyPointCloudFiles['outerPointCloud'])
        myOuterPointCloud = np.where(myOuterPointCloud > 0, 255, 0)

        patientFeatureDict = extractFeaturesFromPointClouds(myInnerPointCloud, myOuterPointCloud, renalArteryDict[patientID], thicknessDict[patientID], pixelSpacingDict[patientID])
        np.save(dictPath + patientID + 'featureDict' + str(datetime.datetime.today()).split('.')[0].replace(':', '-') + '.npy', patientFeatureDict)

def turnFeatureDictsToDataFrame():
    dictPath = 'C:/Users/Luke/Documents/sharedFolder/4YP/dicts'
    listOfDics = [np.load(dictPath + dict).item() for dict in os.listdir(dictPath)]
    listOfKeysOfDics = [set(dict.item().keys()) for dict in listOfDics]
    uniqueKeys = set.intersection(listOfKeysOfDics)
    print('hi')

if __name__ == "__main__":
    main()
    #turnFeatureDictsToDataFrame()




