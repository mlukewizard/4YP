import os
from scipy import misc
import sys
import numpy as np
import matplotlib.pyplot as plt

def compareSegmentationsFromNumpyAndFolder(input1, input2):
    for input in [input1, input2]:
        if '.npy' in input:
            numpyArrayFolder = input
        else:
            folder = input
    numpySegmentation = np.load(numpyArrayFolder)
    folderSegmentations = []
    for file in sorted(os.listdir(folder)):
        folderSegmentations.append(misc.imread(folder + file, flatten=True))
    diceCoefficients = []
    for i, segmentation1 in enumerate(folderSegmentations):
        segmentation1 = np.where(segmentation1 > 100, 255, 0)
        segmentation2 = numpySegmentation[i, :, :]
        segmentation1 = np.asarray(segmentation1).astype(np.bool)
        segmentation2 = np.asarray(segmentation2).astype(np.bool)
        if i > 200:
            print('ho')
        thisDice = diceCoefficient(segmentation1, segmentation2)
        if np.sum(segmentation1) + np.sum(segmentation2) > 0 and thisDice > 0:
            diceCoefficients.append(diceCoefficient(segmentation1, segmentation2))
    print('Average dice is ' + str(sum(diceCoefficients)/len(diceCoefficients)))
def compareSegmentationsFromFolders(folder1, folder2, innerFolder1 = 'NaN', innerFolder2 = 'NaN'):
    files1 = sorted(os.listdir(folder1))
    files2 = sorted(os.listdir(folder2))
    if len(files1) != len(files2):
        sys.exit('The number of files in your folders isnt the same!')

    images1 = []
    images2 = []
    for file1, file2 in zip(files1, files2):
        image1 = misc.imread(folder1 + file1, flatten=True)
        image2 = misc.imread(folder2 + file2, flatten=True)
        image1 = np.where(image1 > 0, 255, 0)
        image2 = np.where(image2 > 0, 255, 0)
        images1.append(image1)
        images2.append(image2)
    if innerFolder1 != 'NaN' and innerFolder2 != 'NaN':
        files1 = sorted(os.listdir(innerFolder1))
        files2 = sorted(os.listdir(innerFolder2))
        for i, file1, file2 in zip(range(len(files1)), files1, files2):
            image1 = misc.imread(innerFolder1 + file1, flatten=True)
            image2 = misc.imread(innerFolder2 + file2, flatten=True)
            images1[i] = np.where(images1[i] > 0, 255, 0)
            images2[i] = np.where(images2[i] > 0, 255, 0)
            images1[i] = np.subtract(images1[i], image1)
            images2[i] = np.subtract(images2[i], image2)
    dices = []
    weightings = []
    for i, image1, image2 in zip(range(len(images1)), images1, images2):
        image1 = np.asarray(image1).astype(np.bool)
        image2 = np.asarray(image2).astype(np.bool)
        weightings.append(0.5*(np.sum(image1) + np.sum(image2)))
        dices.append(diceCoefficient(image1, image2))
        #print('Dice is ' + str(dices[-1]) + ' Weight is ' + str(weightings[-1]))
    weightings2 = []
    dices2 = []
    for i in range(len(dices)):
        if dices[i] != float('nan') and dices[i] != 0 and weightings[i] != 0 and weightings[i] != float('nan'):
            weightings2.append(weightings[i])
            dices2.append(dices[i])
    weightings = weightings2
    dices = dices2
    weightings = np.asarray(weightings) / np.mean(np.asarray(weightings))
    weightedDices = np.asarray(weightings) * np.asarray(dices)
    print('Average weighted dice is ' + str(np.average(weightedDices)))

def CompareSegmentationsFromImageLists(imageList1, imageList2):
    for image1, image2 in zip(imageList1, imageList2):
        print(diceCoefficient(image1, image2))

def diceCoefficient(image1, image2):
    return 2 * np.sum(np.logical_and(image1, image2))/(np.sum(image1) + np.sum(image2))

def main():
    compareSegmentationsFromNumpyAndFolder('C:/Users/Luke/Documents/sharedFolder/4YP/viewSegmentations/MHThickOuterPointCloud.npy', 'C:/Users/Luke/Documents/sharedFolder/4YP/algoSegmentations/MH_predictions/outerPredictions/')
    #compareSegmentationsFromFolders('C:/Users/Luke/Documents/sharedFolder/4YP/Images/Luke_MH/preAugmentation/outerBinary/',
    #                                'C:/Users/Luke/Documents/sharedFolder/4YP/algoSegmentations/MH_predictions/outerPredictions/',
    #                                innerFolder1='C:/Users/Luke/Documents/sharedFolder/4YP/Images/Luke_MH/preAugmentation/innerBinary/',
    #                                innerFolder2='C:/Users/Luke/Documents/sharedFolder/4YP/algoSegmentations/MH_predictions/innerPredictions/')

if __name__ == "__main__":
    main()