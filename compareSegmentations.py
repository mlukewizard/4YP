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
def compareSegmentationsFromFolders(folder1, folder2):
    files1 = sorted(os.listdir(folder1))
    files2 = sorted(os.listdir(folder2))
    dices = []
    if len(files1) != len(files2):
        sys.exit('The number of files in your folders isnt the same!')
    for file1, file2 in zip(files1, files2):
        image1 = misc.imread(folder1 + file1, flatten=True)
        image2 = misc.imread(folder2 + file2, flatten=True)
        image1 = np.asarray(image1).astype(np.bool)
        image2 = np.asarray(image2).astype(np.bool)
        dices.append(diceCoefficient(image1, image2))
        print(dices[-1])

def CompareSegmentationsFromImageLists(imageList1, imageList2):
    for image1, image2 in zip(imageList1, imageList2):
        print(diceCoefficient(image1, image2))

def diceCoefficient(image1, image2):
    return 2 * np.sum(np.logical_and(image1, image2))/(np.sum(image1) + np.sum(image2))

def main():
    compareSegmentationsFromNumpyAndFolder('C:\\Users\\Luke\\Documents\\sharedFolder\\4YP\\viewSegmentations\\MHThickOuterPointCloud.npy', 'C:\\Users\\Luke\\Documents\\sharedFolder\\4YP\\algoSegmentations\\MH_predictions\\outerPredictions\\')

if __name__ == "__main__":
    main()