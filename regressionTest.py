import numpy as np
import os
import sys
import csv
from random import *
import copy
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import *
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import random
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

def randomForestRegression(dataList, transformer, trueYDataFrame):
    regr =RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
           max_features='auto', max_leaf_nodes=None,
           min_impurity_decrease=0.0, min_impurity_split=None,
           min_samples_leaf=1, min_samples_split=2,
           min_weight_fraction_leaf=0.0, n_estimators=1000, n_jobs=1,
           oob_score=False, random_state=0, verbose=0, warm_start=False)
    testRegressor(regr, dataList, transformer, trueYDataFrame)

def gaussianRegression(dataList, transformer, trueYDataFrame):
    myKernel = RBF(length_scale=1.0, length_scale_bounds=(0.1, 10.0))
    regr = GaussianProcessRegressor(kernel=None, alpha=1e-10, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=100, normalize_y=False, copy_X_train=True, random_state=None)
    testRegressor(regr, dataList, transformer, trueYDataFrame)


def svmRegression(dataList, transformer, trueYDataFrame):
    regr = SVR(kernel='rbf', degree=10, gamma='auto', coef0=0.0, tol=0.0000001, C=100000, epsilon=0.0000001, shrinking=False, cache_size=200, verbose=False, max_iter=-1)
    testRegressor(regr, dataList, transformer, trueYDataFrame)


def mlpRegressor(dataList, transformer, trueYDataFrame):
    regr = MLPRegressor(hidden_layer_sizes=(100, ), activation='relu', solver ='adam', alpha = 0.0001, batch_size ='auto', learning_rate ='constant',
    learning_rate_init = 0.001, power_t = 0.5, max_iter = 200, shuffle = True, random_state = None, tol = 0.0001, verbose = False, warm_start = False,
    momentum = 0.9, nesterovs_momentum = True, early_stopping = False, validation_fraction = 0.1, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-08)
    testRegressor(regr, dataList, transformer, trueYDataFrame)

def createData():
    featureNum = 20
    randomFunction = np.random.rand(featureNum)-0.4
    XDataFrame = np.random.rand(40, featureNum)
    trueYDataFrame = np.zeros(XDataFrame.shape[0])
    for i in range(XDataFrame.shape[0]):
        trueYDataFrame[i] = np.dot(XDataFrame[i, :], randomFunction) #+ 2*(random.random()-0.9)

    listOfData, YTransformer = normaliseDataFrames(XDataFrame, trueYDataFrame)
    return listOfData, YTransformer, trueYDataFrame

def testRegressor(myRegressor, dataList, transformer, trueYDataFrame):
    print(myRegressor)
    points = []
    error = 0
    for data in dataList:
        myRegressor.fit(data['XTrain'], data['YTrain'])
        prediction = myRegressor.predict(np.expand_dims(data['XTest'], axis=0))
        print('Predicted: ' + str(prediction) + ' Label: ' + str(data['YTest']))
        points.append([data['YTest'], prediction])
        error += abs(data['YTest'] - prediction)
    print('Total error is ' + str(error))
    points = np.array(points)
    print('Pearson is ' + str(pearsonr(trueYDataFrame, points[:, 1])))
    print('Spearman is ' + str(spearmanr(trueYDataFrame, points[:, 1])))
    points[:, 1] = transformer.inverse_transform(points[:, 1])
    plt.scatter(trueYDataFrame, points[:, 1])
    plt.xlabel('True FMD Value')
    plt.ylabel('Algorithmically predicted FMD Value')
    axes = plt.gca()
    axes.set_xlim([-3, 6])
    axes.set_ylim([-3, 6])
    #plt.plot(np.unique(trueYDataFrame), np.poly1d(np.polyfit(trueYDataFrame, points[:, 1], 1))(np.unique(trueYDataFrame)))
    plt.show()


def normaliseDataFrames(XDataFrame, trueYDataFrame):
    YDataFrame = copy.deepcopy(trueYDataFrame)
    # Scalers the data for each feature
    XTransformer = StandardScaler()
    YTransformer = StandardScaler()
    #XDataFrame = XTransformer.fit_transform(XDataFrame)
    for featureNum in range(int(XDataFrame.shape[1])):
        XDataFrame[:, featureNum] = XDataFrame[:, featureNum]/max(XDataFrame[:, featureNum])
        XDataFrame[:, featureNum] = XDataFrame[:, featureNum] - np.mean(XDataFrame[:, featureNum])
    YDataFrame = YTransformer.fit_transform(YDataFrame.reshape(-1, 1))[:, 0]

    # Constructs the training arrays for the n dimensional test
    listOfData = [{'XTest': XDataFrame[i, :], 'YTest': YDataFrame[i], 'XTrain': np.concatenate([XDataFrame[0:i, :], XDataFrame[i + 1:, :]]),
                   'YTrain': np.concatenate([YDataFrame[0:i], YDataFrame[i + 1:]])} for i in range(XDataFrame.shape[0] - 1)]
    listOfData.append({'XTest': XDataFrame[-1, :], 'YTest': YDataFrame[-1], 'XTrain': XDataFrame[0:-1, :], 'YTrain': YDataFrame[0:-1]})
    return listOfData, YTransformer


def turnFeatureDictsToDataFrame():
    dictPath = 'C:/Users/Luke/Documents/sharedFolder/4YP/dicts/'
    TPThreeDict = {'AD': -1.944, 'AA': -2.47, 'CC': 0.84, 'FS': -1.944, 'CE': 1.1173, 'FW': 0.96, 'CG': 2.8759, 'CI': 2.0778, 'CK': 1.0058, 'GA': -0.19, 'YC': 4.7557,
                   'CM': 2.0638, 'CO': -0.812, 'CQ': -0.79, 'CS': 4.9854, 'CU': -0.39, 'CW': 0.4763, 'XC': 1.3208, 'GQ': 2.9062, 'DK': 0, 'AG': 0.3556, 'DM': 0.3556,
                   'DQ': 1.07, 'GY': 0.6962, 'AJ': 0.2899, 'HC': 0.2899, 'HG': 1.9501, 'DW': 1.7311, 'HI': 1.6085, 'EI': 1.7013, 'EK': 1.875, 'EO': 0.88, 'EQ': 1.2077,
                   'ES': -1.03092783505154, 'EU': 0.59642147117295, 'EY': 4.3716, 'FA': -2.105, 'HS': 3.55029585798817, 'HW': 1.42857142857142, 'FG': 2.13333333333334,
                   'FI': -0.544959128065396}
    dictOfDics = {}
    for dict in sorted(os.listdir(dictPath)):
        dictOfDics[dict[0:2]] = np.load(dictPath + dict).item()
    XDataFrame = []
    trueYDataFrame = []
    for key, FMDVal in TPThreeDict.items():
        if key in dictOfDics.keys():
            featureDict = dictOfDics[key]
            YRow = FMDVal
            XRow =[np.nanmax(featureDict['innerDiameter']),  #0
                   np.nanmax(featureDict['outerDiameter']),  #1
                   np.nanmax(featureDict['innerPerimeter']),  #2
                   np.nanmax(featureDict['outerPerimeter']), #3

                   np.nanmax(featureDict['innerArea']), #4
                   np.nanmax(featureDict['outerArea']), #5
                   featureDict['innerAreaDistal'],  #6
                   featureDict['innerAreaProximal'],  #7
                   featureDict['outerAreaDistal'],  #8
                   featureDict['outerAreaProximal'],  #9

                   featureDict['outerAorticVolume'],  # 10
                   featureDict['innerAorticVolume'],  # 11
                   featureDict['innerNeckVolume'],  # 12
                   featureDict['outerNeckVolume'],  # 13
                   featureDict['AAAThrombusVolume'],  # 14
                   featureDict['neckThrombusVolume'],  # 15

                   featureDict['hNeck'],  #16
                   featureDict['hSac'],  #17
                   featureDict['lNeck'],  #18
                   featureDict['lSac'],  #19
                   featureDict['bulgeHeight'], #20

                   np.nanmean(featureDict['AAAInnerTortuosityLargeScale']),  #21
                   np.nanmean(featureDict['AAAInnerTortuositySmallScale']),  #22
                   np.nanmean(featureDict['AAAOuterTortuosityLargeScale']),  #23
                   np.nanmean(featureDict['AAAOuterTortuositySmallScale']),  #24
                   np.nanmax(featureDict['AAAInnerTortuosityLargeScale']),  #25
                   np.nanmax(featureDict['AAAInnerTortuositySmallScale']),  #26
                   np.nanmax(featureDict['AAAOuterTortuosityLargeScale']),  #27
                   np.nanmax(featureDict['AAAOuterTortuositySmallScale'])  #28
                   ]
            XDataFrame.append(XRow)
            trueYDataFrame.append(YRow)
        else:
            print('Couldnt find patient ' + key + ' you should find where this has gone')
    XDataFrame = np.array(XDataFrame)
    trueYDataFrame = np.array(trueYDataFrame)

    listOfData, YTransformer = normaliseDataFrames(XDataFrame, trueYDataFrame)
    return listOfData, YTransformer, trueYDataFrame

def randomResults():
    dictPath = 'C:/Users/Luke/Documents/sharedFolder/4YP/dicts/'
    TPThreeDict = {'AD': -1.944, 'AA': -2.47, 'CC': 0.84, 'FS': -1.944, 'CE': 1.1173, 'FW': 0.96, 'CG': 2.8759, 'CI': 2.0778, 'CK': 1.0058, 'GA': -0.19, 'YC': 4.7557,
                   'CM': 2.0638, 'CO': -0.812, 'CQ': -0.79, 'CS': 4.9854, 'CU': -0.39, 'CW': 0.4763, 'XC': 1.3208, 'GQ': 2.9062, 'DK': 0, 'AG': 0.3556, 'DM': 0.3556,
                   'DQ': 1.07, 'GY': 0.6962, 'AJ': 0.2899, 'HC': 0.2899, 'HG': 1.9501, 'DW': 1.7311, 'HI': 1.6085, 'EI': 1.7013, 'EK': 1.875, 'EO': 0.88, 'EQ': 1.2077,
                   'ES': -1.03092783505154, 'EU': 0.59642147117295, 'EY': 4.3716, 'FA': -2.105, 'HS': 3.55029585798817, 'HW': 1.42857142857142, 'FG': 2.13333333333334,
                   'FI': -0.544959128065396}
    dictOfDics = {}
    for dict in sorted(os.listdir(dictPath)):
        dictOfDics[dict[0:2]] = np.load(dictPath + dict).item()
    XDataFrame = []
    trueYDataFrame = []
    for key, FMDVal in TPThreeDict.items():
        if key in dictOfDics.keys():
            featureDict = dictOfDics[key]
            YRow = FMDVal
        trueYDataFrame.append(YRow)
    trueYDataFrame = np.array(trueYDataFrame)
    randomYDataFrame = np.random.rand(trueYDataFrame.shape[0])
    plt.scatter(trueYDataFrame, randomYDataFrame)
    plt.plot(np.unique(trueYDataFrame), np.poly1d(np.polyfit(trueYDataFrame, randomYDataFrame, 1))(np.unique(trueYDataFrame)))
    plt.show()

def main():
    data, transformer, trueYDataFrame = turnFeatureDictsToDataFrame()
    #data, transformer, trueYDataFrame = createData()

    #randomForestRegression(data, transformer, trueYDataFrame)
    print()
    print()
    gaussianRegression(data, transformer, trueYDataFrame)
    print()
    print()
    #svmRegression(data, transformer, trueYDataFrame)
    print()
    print()
    #mlpRegressor(data, transformer, trueYDataFrame)

if __name__ == '__main__':
    main()
    #randomResults()
