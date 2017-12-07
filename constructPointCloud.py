from __future__ import print_function
import os, shutil
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

innerBinaryDir = '/media/sf_sharedFolder/Images/NS/preAugmentation/innerBinary/'
outerBinaryDir = '/media/sf_sharedFolder/Images/NS/preAugmentation/outerBinary/'

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

    #plt.imshow(imageHolder, cmap='gray')
    #plt.show()
    innerPointCloud[fileNum, :, :] = imageHolder
    print('Completed ' + str(fileNum) + ' of ' + str(height))


plt.figure()
z,x,y = innerPointCloud.nonzero()
ax = plt.subplot(111, projection='3d')
ax.scatter(x, y, -z, zdir = 'z', c = 'red')
plt.show()
