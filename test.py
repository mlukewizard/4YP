from scipy import misc
from scipy import ndimage
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import matplotlib.pyplot as plt
from myFunctions import *
import copy
import sys

#DC
#file = 'C:\\Users\\Luke\\Documents\\sharedFolder\\4YP\\Images\\Regent_DC\\preAugmentation\\innerBinary\\inner109Binary.png'
dicomFile = 'C:\\Users\\Luke\\Documents\\sharedFolder\\4YP\\Images_256_ShelledOuter\\Regent_DC\\postAugmentation\\croppedDicoms\\Augment01Original223PatientDC.png'
binaryFile = 'C:\\Users\\Luke\\Documents\\sharedFolder\\4YP\\Images_256_ShelledOuter\\Regent_DC\\postAugmentation\\outerAugmented\\Augment01OuterBinary223PatientDC.png'

dicomImage = misc.imread(dicomFile, flatten=True)
plt.subplot(2, 2, 1)
plt.imshow(dicomImage, cmap='gray')
dicomImage = lukesImageDiverge(dicomImage, [73, 41], -15)
plt.subplot(2, 2, 2)
plt.imshow(dicomImage, cmap='gray')

binaryImage = misc.imread(binaryFile, flatten=True)
plt.subplot(2, 2, 3)
plt.imshow(binaryImage, cmap='gray')
binaryImage = lukesImageDiverge(binaryImage, [73, 41], -15)
binaryImage = binaryImage[binaryImage > 0] == 255
plt.subplot(2, 2, 4)
plt.imshow(binaryImage, cmap='gray')
plt.show()