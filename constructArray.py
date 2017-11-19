import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import scipy
from scipy import misc

imageDirectory = '/media/sf_sharedFolder/Images/39894NS/PostAugmentation/nonAugmentedInnerOriginals/'
arrayDirectory = '/media/sf_sharedFolder/npArrays'
patientID = 'NS'
imageType = 'Original'

fileList = sorted(os.listdir(imageDirectory))
imgTotal = len(fileList)
totalCounter = 0
maxSliceNum = 475
binNum = 1

for fileString in fileList:
    largestNumInString = findLargestNumber(fileString)
    if largestNumInString > maxSliceNum:
        maxSliceNum = largestNumInString

npImageArray = np.ndarray((binNum * maxSliceNum, 512, 512, 1), dtype='float32')

print('Loop starting')
for filename in fileList:

    split1 = filename.split(imageType)
    split2 = split1[0].split('Augment')
    augNum = int(split2[1])
    split3 = split1[1].split('Patient')
    sliceNum = int(split3[0])

    arrayIndex = (sliceNum-1) + ((augNum%10 if augNum%10 < 6 else (augNum%10) -5)-1)*maxSliceNum
    image = misc.imread(imageDirectory + filename)
    #print(image[250,250])
    #plt.imshow(image)
    #plt.show()
    npImageArray[arrayIndex, :, :, 0] = image
    totalCounter = totalCounter + 1

    if ((augNum % 10 == 0) or (augNum % 10 == 5)) and (sliceNum == maxSliceNum):
	print('Saved one')
        np.save(arrayDirectory + 'Augment' + "%03d" % (augNum-4) + '-' + "%03d" % (augNum) + 'Patient' + patientID + '_' + imageType + '.npy', npImageArray)

