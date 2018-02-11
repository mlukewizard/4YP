from __future__ import division
from __future__ import print_function
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import scipy
from scipy import misc
import math
import os
import copy

def trainModel(patientList, trainingArrayDepth, twoDVersion, boxSize, dicomFileList, trainingArrayPath, validationArrayPath, model_folder, img_test_file, bm_test_file):
    from keras import backend as K
    K.clear_session()
    from keras.callbacks import ModelCheckpoint
    from keras.models import Model, load_model
    from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, ConvLSTM2D, LSTM, TimeDistributed, \
        Bidirectional, Dropout, BatchNormalization, Activation
    from keras.optimizers import Adam
    from keras import losses
    import numpy as np
    import os
    import sys
    import h5py
    from keras import optimizers
    from random import uniform, shuffle
    from myModels import my3DModel, my2DModel
    # Defining loss only for the middle slice (if needed)
    def my_loss(y_true, y_pred):
        return K.mean(K.binary_crossentropy(y_true[:, 2, :, :, :], y_pred[:, 2, :, :, :]))
    losses.my_loss = my_loss

    # Loading numpy arrays for validation
    img_test = np.load(os.path.join(validationArrayPath, img_test_file))
    bm_test = np.load(os.path.join(validationArrayPath, bm_test_file)) / 255

    # Initilising training arrays
    if not twoDVersion:
        img_measure = np.ndarray((trainingArrayDepth, 5, boxSize, boxSize, 1), dtype='float32')
        bm_measure = np.ndarray((trainingArrayDepth, 5, boxSize, boxSize, 2), dtype='float32')
    else:
        img_measure = np.ndarray((trainingArrayDepth, boxSize, boxSize, 1), dtype='float32')
        bm_measure = np.ndarray((trainingArrayDepth, boxSize, boxSize, 2), dtype='float32')

    print('Constructing Arrays')
    # Defining the split of slices from each of the randomly selected arrays
    arraySplits = np.linspace(0, trainingArrayDepth, len(patientList) + 1, dtype='int')

    for i in range(len(arraySplits) - 1):
        # This while loop ensures you get a patient from each patient for each epoch
        if not dicomFileList:
            print('Refreshing dicomFileList')
            dicomFileList = filter(lambda k: 'dicom' in k, sorted(os.listdir(trainingArrayPath)))
        patientFile = 'RandomString'
        while patientFile.find(patientList[i]) == -1:
            shuffle(dicomFileList)
            patientFile = dicomFileList[0]
        print('Using data from ' + patientFile)
        dicomFileList.pop(0)

        # Loads arrays of images
        dicomFile = np.load(trainingArrayPath + patientFile)
        binaryFile = np.load(trainingArrayPath + patientFile.split("dic")[0] + 'binary.npy') / 255

        # Randomly writes to the training arrays from the contents of the arrays of images
        for j in range(arraySplits[i + 1] - arraySplits[i]):
            index = int(np.round(uniform(0, len(dicomFile) - 1)))
            if not twoDVersion:
                img_measure[arraySplits[i] + j, :, :, :, :] = dicomFile[index]
                bm_measure[arraySplits[i] + j, :, :, :, :] = binaryFile[index]
            else:
                img_measure[arraySplits[i] + j, :, :, :] = dicomFile[index]
                bm_measure[arraySplits[i] + j, :, :, :] = binaryFile[index]

    # Defines the test split so that your validtion array doesnt feature in the training set
    testSplit = img_test.shape[0] / (img_test.shape[0] + img_measure.shape[0])
    print('Validation split is ' + str(testSplit))

    # Concatenates the two arrays
    # img_train = np.concatenate((img_measure, img_test))
    # bm_train = np.concatenate((bm_measure, bm_test))

    print('Building Model')
    model_list = os.listdir(model_folder)  # Checking if there is an existing model
    if model_list.__len__() == 0:  # Creating a new model if empty

        # Get the model you want to use from the models bank
        if not twoDVersion:
            model = my3DModel(boxSize)
        else:
            model = my2DModel(boxSize)

        # If there isnt a previous number then this epoch must be epoch number 0
        epoch_number = 0

    else:
        # Scrolls through the model list and find the model with the highest epoch number
        currentMax = 0
        for fn in model_list:
            epoch_number = int(fn.split('weights.')[1].split('-')[0])
            if epoch_number > currentMax:
                currentMax = epoch_number
                model_file = fn
        epoch_number = int(model_file.split('weights.')[1].split('-')[0])

        # Loads that model file
        print('Using model: ' + model_folder + model_file)
        f_model = h5py.File(os.path.join(model_folder, model_file), 'r+')
        if 'optimizer_weights' in f_model:
            del f_model['optimizer_weights']
        f_model.close()

        # Loads the model from that file
        model = load_model(os.path.join(model_folder, model_file))
        print('Using model number ' + str(epoch_number))

    # Defines the compile settings
    if not twoDVersion:
        model.compile(optimizer=Adam(lr=7e-4), loss=my_loss)
    else:
        model.compile(optimizer=Adam(lr=1e-3), loss=losses.binary_crossentropy)

    # Defines the checkpoint file
    model_check_file = os.path.join(model_folder, 'weights.{epoch:02d}-{loss:.2f}.h5')
    model_checkpoint = ModelCheckpoint(model_check_file, monitor='val_loss', save_best_only=False)

    # Actually do the training for this epoch
    print('Starting train')
    if not twoDVersion:
        myBatchSize = 2
    else:
        myBatchSize = 4
    model.fit(np.concatenate((img_measure, img_test)), np.concatenate((bm_measure, bm_test)), batch_size=myBatchSize,
              initial_epoch=epoch_number, epochs=epoch_number + 1, verbose=1, shuffle=True, validation_split=testSplit,
              callbacks=[model_checkpoint])
    return dicomFileList


def lukesImageDiverge(image, divergePoint, divergeFactor):
    #NB. diverge point is in [x ,y] or colNum, rowNum
    newImage = copy.deepcopy(image)
    getCoefficient = lambda x: x/(2.5*abs(divergeFactor)) if abs(x/(2.5*abs(divergeFactor))) < 1 else 1
    maxDistortionDistance = abs(int(2.5*divergeFactor))
    if max(divergePoint) + maxDistortionDistance > image.shape[0] or min(divergePoint) - maxDistortionDistance < 0:
        sys.exit('Youre trying to augment out of the image range')
    for i in range(divergePoint[1] - maxDistortionDistance, divergePoint[1] + maxDistortionDistance):
        for j in range(divergePoint[0] - maxDistortionDistance, divergePoint[0] + maxDistortionDistance):
            xDist = j - divergePoint[0]
            yDist = i - divergePoint[1]
            multiplier = getCoefficient(np.sqrt(np.square(xDist) + np.square(yDist)))
            if divergeFactor > 0:
                newImage[i, j] = image[int(i - yDist * (1 - multiplier)), int(j - xDist * (1 - multiplier))]
            if divergeFactor < 0:
                newImage[i, j] = image[int(i + yDist * (1 - multiplier)), int(j + xDist * (1 - multiplier))]
    return newImage

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

def findAAABounds(wallVolume, OuterDiameter):
    maxDia = len(OuterDiameter[0:-100]) + np.argmax(OuterDiameter[-100:])
    maxVol = len(wallVolume[0:-100]) + np.argmax(wallVolume[-100:])
    acceptableSteps = np.linspace(1, 50, 50, dtype='int').tolist()
    timeSerieses = [wallVolume, OuterDiameter, wallVolume, OuterDiameter]
    directions = [-1, -1, 1, 1]
    points = []
    maxLocations = [maxVol, maxDia, maxVol, maxDia]
    for timeSeries, direction, start in zip(timeSerieses, directions, maxLocations):
        looking = True
        copyStart = copy.deepcopy(start)
        while looking:
            if any(((np.append(timeSeries, np.zeros(60, dtype = 'int'))[copyStart + direction*step] - np.append(timeSeries, np.zeros(60, dtype = 'int'))[copyStart])/step < -np.max(timeSeries)/200 and np.append(timeSeries, np.zeros(60, dtype = 'int'))[copyStart + direction*step] != 0) for step in acceptableSteps):
                copyStart = copyStart + direction
            else:
                looking = False
        points.append(copyStart)
    if abs(points[3] - points[2]) > 10:
        print('Warning: Your algorithm is unsure about where the aneurysm ends, wall thickness suggests ' + str(points[2]) + ' and diameter suggests ' + str(points[3]))
    if abs(points[1] - points[0]) > 10:
        print('Warning: Your algorithm is unsure about where the aneurysm starts, wall thickness suggests ' + str(points[0]) + ' and diameter suggests ' + str(points[1]))
    return {'AAAStart':int((points[0] + points[1])/2), 'AAAEnd':int((points[2] + points[3])/2)}
    #return [int((points[0] + points[1])/2), int((points[2] + points[3])/2)]

def isDoubleAAA(image):
    if np.max(image) != 255:
        return False
    OuterTop = np.array(np.where(np.isin(image, 255)))[:, 0]
    OuterBottom = np.array(np.where(np.isin(image, 255)))[:, -1]
    continueLoop = True
    while continueLoop:
        x = (OuterTop - OuterBottom)/np.max(abs((OuterBottom - OuterTop)))
        if image[OuterBottom[0] + int(x[0]), OuterBottom[1] + int(x[1])] == 255 or image[OuterBottom[0] + 2*int(x[0]), OuterBottom[1] + 2*int(x[1])] == 255 or image[OuterBottom[0] + 5*int(x[0]), OuterBottom[1] + 5*int(x[1])] == 255:
            OuterBottom[0] = OuterBottom[0] + int(x[0])
            OuterBottom[1] = OuterBottom[1] + int(x[1])
        else:
            return True
        if np.square(OuterTop[0] - OuterBottom[0]) + np.square(OuterTop[1] - OuterBottom[1]) < 400:
            return False

def ConstructArraySlice(inputFolder1, inputFolder1Dir, inputFileIndex, boxSize,inputFolder2=None, inputFolder2Dir= 'Blank', centralLocation=None, twoDVersion = False):
    import scipy.misc as misc
    import sys
    import os
    import dicom
    import PIL
    from PIL import Image
    import matplotlib.pyplot as plt
    from uuid import getnode as get_mac
    mac = get_mac()
    if mac != 176507742233701:
        tmpFolder = '/home/lukemarkham1383/trainEnvironment/4YP_Python/tmp/'
    else:
        tmpFolder = 'C:\\Users\\Luke\\Documents\\sharedFolder\\4YP\\4YP_Python\\tmp\\'
    if not os.path.exists(tmpFolder):
        os.mkdir(tmpFolder)
    dicomSlice = False

    # Just a little check
    if inputFolder2!= None and len(inputFolder2) != len(inputFolder1):
        sys.exit('Your folders arent the same length, something is seriously wrong')

    # Making the distinction between binary and dicom a little easier
    if inputFolder2 == None:
        dicomSlice = True

    if dicomSlice == True:
        fileLists = [inputFolder1]
        image1Files = []
        doc = [image1Files]
    else:
        image1Files = []
        image2Files = []
        fileLists = [inputFolder1, inputFolder2]
        doc = [image1Files, image2Files]

    if not twoDVersion:
        arrayIndexes = np.linspace(inputFileIndex-6, inputFileIndex+6, 5, dtype='int')
        for i in range(len(arrayIndexes)):
            if arrayIndexes[i] < 0:
                arrayIndexes[i] = 0
            elif arrayIndexes[i] > len(inputFolder1)-1:
                arrayIndexes[i] = len(inputFolder1)-1
    else:
        arrayIndexes = [inputFileIndex]
    #print(arrayIndexes)
    for imageFiles, inputFolder, fileList, inputFolderDir in zip(doc, [inputFolder1, inputFolder2], fileLists, [inputFolder1Dir, inputFolder2Dir]):
        for i in range(len(arrayIndexes)):
            if '.' in fileList[arrayIndexes[i]]:
                imageFiles.append(np.array(Image.open(inputFolderDir + fileList[arrayIndexes[i]]).convert('F')))
            else:
                dicomImage = dicom.read_file(inputFolderDir + fileList[arrayIndexes[i]]).pixel_array
                misc.imsave(tmpFolder + 'dicomTemp.png', dicomImage)
                dicomImage = misc.imread(tmpFolder + 'dicomTemp.png', flatten=True)
                # plt.imshow(dicomImage, cmap='gray')
                # plt.show()
                dicomImage = lukesAugment(dicomImage)
                imageFiles.append(dicomImage)
                os.remove(tmpFolder + 'dicomTemp.png')
        if (not imageFiles[0].shape[0] == boxSize or not imageFiles[0].shape[1] == boxSize) and (centralLocation is None):
            sys.exit('If your image isnt 144x144 then you need to tell me the central location')
        elif (not imageFiles[0].shape[0] == boxSize or not imageFiles[0].shape[1] == boxSize) and (centralLocation is not None):
            upperRow = centralLocation[0] - round(boxSize/2)
            lowerRow = upperRow + boxSize
            leftColumn = centralLocation[1] - round(boxSize/2)
            rightColumn = leftColumn + boxSize
            for i in range(len(arrayIndexes)):
                imageFiles[i] = imageFiles[i][upperRow:lowerRow, leftColumn:rightColumn]

    if not twoDVersion:
        if dicomSlice:
            slice = np.ndarray((len(arrayIndexes), boxSize, boxSize, 1), dtype='float32')
            for i in range(len(arrayIndexes)):
                slice[i, :, :, 0] = image1Files[i]
            return slice
        else:
            slice = np.ndarray((len(arrayIndexes), boxSize, boxSize, 2), dtype='float32')
            for i in range(len(arrayIndexes)):
                slice[i, :, :, 0] = image1Files[i]
                slice[i, :, :, 1] = image2Files[i]
            #saveSlice(slice, showFig=True)
            return slice
    else:
        if dicomSlice:
            slice = np.ndarray((boxSize, boxSize, 1), dtype='float32')
            slice[:, :, 0] = image1Files[i]
            return slice
        else:
            slice = np.ndarray((boxSize, boxSize, 2), dtype='float32')
            slice[:, :, 0] = image1Files[i]
            slice[:, :, 1] = image2Files[i]
            return slice

def findLargestNumberInString(text):
    li = [0]
    for i in range(len(text)):
        num = ""
        if text[i].isdigit():
            while text[i].isdigit():
                num = num + text[i]
                i = i + 1
            li.append(int(num))
    return max(li)

def findLargestNumberInFolder(list):
    import types
    largestNum = 0
    for i in range(len(list)):
        if (findLargestNumberInString(list[i]) > largestNum):
            largestNum = findLargestNumberInString(list[i])
    return largestNum

def findSmallestNumberInString(text):
    li = []
    for i in range(len(text)):
        num = ""
        if text[i].isdigit():
            while text[i].isdigit():
                num = num + text[i]
                i = i + 1
            li.append(int(num))
    return min(li)

def findSmallestNumberInFolder(list):
    smallestNum = 10000
    for i in range(len(list)):
        if (findSmallestNumberInString(list[i]) < smallestNum):
            smallestNum = findSmallestNumberInString(list[i])
    return smallestNum

def getImagePerimeterPoints(inputImage):
    image = Image.fromarray(inputImage)
    image = image.filter(ImageFilter.FIND_EDGES)
    outputImage = np.array(image)
    return outputImage

def getImageEdgeCoordinates(inputImage, edge):
    import numpy as np
    import sys
    import copy

    workingImage = copy.deepcopy(inputImage)
    if edge == 'Left':
        return np.array([np.where(np.transpose(workingImage) > 10)[0][0], np.where(np.transpose(workingImage) > 10)[1][0]])
    elif edge == 'Right':
        return np.array([np.where(np.transpose(workingImage) > 10)[0][-1], np.where(np.transpose(workingImage) > 10)[1][-1]])
    elif edge == 'Top':
        return np.array([np.where((workingImage) > 10)[1][0], np.where((workingImage) > 10)[0][0]])
    elif edge == 'Bottom':
        return np.array([np.where((workingImage) > 10)[1][-1], np.where((workingImage) > 10)[0][-1]])
    elif edge == 'Center':
        return np.array([int((np.where(np.transpose(workingImage) > 10)[0][0] + np.where(np.transpose(workingImage) > 10)[0][-1])/2), int((np.where((workingImage) > 10)[0][-1] + np.where((workingImage) > 10)[0][0])/2)])
    else:
        sys.exit('The edge type specified was not valid')

def lukesBinarize(inputImage):
    import numpy as np
    if np.max(inputImage) > 210:
        idx = inputImage > 210
        inputImage[idx] = 255
    idx = inputImage < 210
    inputImage[idx] = 0
    return inputImage

def getImageBoundingBox(inputImage):
    from scipy import ndimage
    import numpy as np

    if np.max(inputImage) > 0:
        #Order is topY, bottomY, leftX, rightX
        return np.array([min(np.where(np.transpose(inputImage) > 10)[0]), max(np.where(inputImage > 10)[1]), min(np.where(inputImage > 10)[0]), max(np.where(np.transpose(inputImage) > 10)[1])])
    else:
        return np.array([256, 256, 256, 256])

def getFolderBoundingBox(filePath):
    import os
    import numpy as np
    from scipy import misc
    import matplotlib.pyplot as plt
    boxSize = 512
    cumulativeImage = np.zeros(shape=(512,512), dtype='float32')
    fileList = sorted(os.listdir(filePath))

    for filename in fileList:
        newFile = np.array(misc.imread(filePath + filename, flatten=True))
        cumulativeImage = np.add(cumulativeImage, newFile)

    #plt.imshow(cumulativeImage, cmap='gray')
    #plt.show()
    return getImageBoundingBox(cumulativeImage)

def getFolderCoM(dicomFolder):
    import dicom
    import os
    from scipy import ndimage
    import numpy as np
    import math
    boxSize = 512

    inputImage = np.ndarray([boxSize, boxSize])
    fileList = sorted(os.listdir(dicomFolder))
    sampleFileList = filter(lambda k: '80' in k, fileList)
    i = 0
    xTotal = 0
    yTotal = 0
    for filename in sampleFileList:
        i = i + 1
        image = dicom.read_file(dicomFolder + filename)
        inputImage[:, :] = image.pixel_array

        # Gets image centre of mass, note y coordinate comes first and then x coordinate
        CoM = ndimage.measurements.center_of_mass(inputImage)
        xTotal = xTotal + CoM[1]
        yTotal = yTotal + CoM[0]

    xAvg = math.floor(xTotal / i)
    yAvg = math.floor(yTotal / i)

    # Sets the limits for a 256x256 bounding box
    xMin = int(xAvg - 128 if xAvg - 128 > 0 else 0)
    xMax = int(xMin + 256)
    yMin = int(yAvg - 128 if yAvg - 128 > 0 else 0)
    yMax = int(yMin + 256)
    return np.array([xMin, xMax, yMin, yMax])

def lukesAugment(image):
    import numpy as np

    '''
    This is the augment for if youve normalised the image with imsave
    def f(x):
            return 135.78315*np.tanh((x-150)/70) + 132.0961
            
    #More relaxed normalisation
    def f(x):
        return 261.0482 + (-3.771016e-15 - 261.0482) / (1 + np.power(x / 74.43762, 3.03866))

    #Aggresive normalisation
    def f(x):
        return 261.3943 + (-1.963209e-15 - 261.3943) / (1 + np.power((x / 69.10387), 2.822998))
    '''

    #Highly aggresive normalisation
    def f(x):
        return 255.4025 + (-7.706867e-15 - 255.4025)/(1 + np.power((x/79.25257), 5.520541))

    f = np.vectorize(f)  # or use a different name if you want to keep the original f
    image = f(image)
    return image

def saveSlice(slice1, slice2=np.zeros(7), saveFig = False, showFig = False, saveFolder = ''):
    import matplotlib.pyplot as plt
    import numpy as np
    import datetime
    import sys
    counter = 0
    mySlices = [slice1]
    if not slice2.shape[0] == 7:
        mySlices.append(slice2)
    totalDepth = 0

    #This loop sums the depths on the slices so we know how big to make the plot
    for slice in mySlices:
        if len(slice.shape) == 4:
            sliceDepth = slice.shape[3]
            totalDepth = totalDepth + sliceDepth
        else:
            sliceDepth = slice.shape[4]
            totalDepth = totalDepth + sliceDepth

    #This loop plots each of the slices
    for slice in mySlices:
        if len(slice.shape) == 4:
            sliceWidth = 1
            sliceDepth = slice.shape[3]
            twoD = True
        else:
            sliceWidth = slice.shape[1]
            sliceDepth = slice.shape[4]
            twoD = False
        for j in range(sliceDepth):
            for i in range(sliceWidth):
                plt.subplot(totalDepth, sliceWidth, counter+1)
                if not twoD:
                    plt.imshow(slice[0, i, :, :, j], cmap='gray')
                else:
                    plt.imshow(slice[0, :, :, j], cmap='gray')
                plt.axis('off')
                counter = counter + 1

    if showFig == True:
        mng = plt.get_current_fig_manager()
        mng.window.state('zoomed')
        plt.show()
    if saveFig == True:
        if saveFolder == '':
            sys.exit('If you set saveFig as true you also need to specify saveFolder')
        else:
            plt.savefig(saveFolder + 'Figure_' + str(datetime.datetime.now()).split('.')[0].replace(':','-').replace(' ','_'))
