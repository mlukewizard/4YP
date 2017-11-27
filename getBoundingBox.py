def getImageBoundingBox(inputImage):
    from scipy import ndimage
    import numpy as np

    image = ndimage.gaussian_filter(inputImage, sigma=1)

    minX = 500
    maxX = 10
    maxY = 10
    minY = 500
    for i in range(image.shape[1]):
        for j in range(image.shape[0]):
            if (image[j,i] > 254):
                if (i < minX):
                    minX = i
                elif (i > maxX):
                    maxX = i
                if (j < minY):
                    minY = j
                elif (j > maxY):
                    maxY = j
    return np.array([minX, maxX, minY, maxY])

def getFolderBoundingBox(filePath):
    import os
    import numpy as np
    from scipy import misc
    cumulativeImage = np.ndarray([512,512])
    fileList = sorted(os.listdir(filePath))
    for filename in fileList:
        cumulativeImage = cumulativeImage + misc.imread(filePath + filename)
    return getImageBoundingBox(cumulativeImage)

print(getFolderBoundingBox('/media/sf_sharedFolder/Images/39894NS/512/PostAugmentation/nonAugmentedInnerBinary/'))
