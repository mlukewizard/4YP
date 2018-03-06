import numpy as np
import os
import sys
import csv
from random import *
import copy
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import random
import matplotlib.pyplot as plt

def randomForestRegression(dataList, transformer):
    print('Predicting using random forest')
    error = 0
    for data in dataList:
        regr =RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
               max_features='auto', max_leaf_nodes=None,
               min_impurity_decrease=0.0, min_impurity_split=None,
               min_samples_leaf=1, min_samples_split=2,
               min_weight_fraction_leaf=0.0, n_estimators=1000, n_jobs=1,
               oob_score=False, random_state=0, verbose=0, warm_start=False)
        regr.fit(data['XTrain'], data['YTrain'])
        prediction = regr.predict(np.expand_dims(data['XTest'], axis=0))
        print('Predicted: ' + str(prediction) + ' Label: ' + str(data['YTest']))
        error += abs(data['YTest'] - prediction)
    print('Total error is ' + str(error))


def gaussianRegression(dataList, transformer, trueYDataFrame):
    points = []
    print('Predicting using gaussian regressor')
    error = 0
    for data in dataList:
        Gregr = GaussianProcessRegressor(kernel=None, alpha=1e-10, n_restarts_optimizer=0, normalize_y=True, copy_X_train=True, random_state=None)

        Gregr.fit(data['XTrain'], data['YTrain'])
        prediction = Gregr.predict(np.expand_dims(data['XTest'], axis=0))
        print('Predicted: ' + str(prediction) + ' Label: ' + str(data['YTest']))
        points.append([data['YTest'], prediction])
        error += abs(data['YTest'] - prediction)
    points = np.array(points)
    print('Total error is ' + str(error))
    points[:, 0] = transformer.inverse_transform(points[:, 0])
    points[:, 1] = transformer.inverse_transform(points[:, 1])
    plt.scatter(trueYDataFrame, points[:, 1])
    plt.show()

def svmRegression(dataList, transformer):
    print('Predicting using svm regressor')
    error = 0
    for data in dataList:
        segr = SVR(kernel='rbf', degree=20, gamma='auto', coef0=0.0, tol=0.001, C=10000, epsilon=0.000001, shrinking=True, cache_size=200, verbose=False, max_iter=-1)

        segr.fit(data['XTrain'], data['YTrain'])
        prediction = segr.predict(np.expand_dims(data['XTest'], axis=0))
        print('Predicted: ' + str(prediction) + ' Label: ' + str(data['YTest']))
        error += abs(data['YTest'] - prediction)
    print('Total error is ' + str(error))


def mlpRegressor(dataList, transformer):
    print('Predicting using multilayer perceptron')
    error = 0
    for data in dataList:
        MLPRRegr = MLPRegressor(hidden_layer_sizes=(100, ), activation='relu', solver ='adam', alpha = 0.0001, batch_size ='auto', learning_rate ='constant',
        learning_rate_init = 0.001, power_t = 0.5, max_iter = 200, shuffle = True, random_state = None, tol = 0.0001, verbose = False, warm_start = False,
        momentum = 0.9, nesterovs_momentum = True, early_stopping = False, validation_fraction = 0.1, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-08)
        MLPRRegr.fit(data['XTrain'], data['YTrain'])
        prediction = MLPRRegr.predict(np.expand_dims(data['XTest'], axis=0))
        print('Predicted: ' + str(prediction) + ' Label: ' + str(data['YTest']))
        error += abs(data['YTest'] - prediction)
    print('Total error is ' + str(error))



def createData():
    randomFunction = np.random.rand(25)
    randomData = np.random.rand(41, 25)
    answers = np.random.rand(randomData.shape[0])
    for i in range(randomData.shape[0]):
        row = randomData[i, :]*randomFunction + 100*random.random()
    XTrain = randomData[0:-5, :]
    XTest = randomData[-5:, :]
    YTrain = answers[0:-5]
    YTest = answers[-5:]
    return {'XTest':XTest, 'YTest':YTest, 'XTrain':XTrain, 'YTrain':YTrain}

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
    dataFrame = []
    for key, FMDVal in TPThreeDict.items():
        if key in dictOfDics.keys():
            featureDict = dictOfDics[key]
            row = [FMDVal,
                   featureDict['maxInnerArea'], #0
                   featureDict['maxOuterArea'], #1
                   featureDict['dInnerAreaDistal'], #2
                   featureDict['dInnerAreaProximal'], #3
                   featureDict['dOuterAreaDistal'], #4
                   featureDict['dOuterAreaProximal'], #5
                   featureDict['hNeck'], #6 is negative sometimes...
                   featureDict['hSac'], #7 is negative sometimes...
                   featureDict['lNeck'], #8 doesnt exist
                   featureDict['lSac'], #9 doesnt exist
                   featureDict['bulgeHeight'], #10
                   featureDict['outerAorticVolume'], #11
                   featureDict['innerAorticVolume'], #12
                   featureDict['innerNeckVolume'], #13
                   featureDict['outerNeckVolume'], #14
                   featureDict['AAAThrombusVolume'], #15
                   featureDict['neckThrombusVolume'], #16
                   np.average(featureDict['AAAInnerTortuosityLargeScale']), #17
                   np.average(featureDict['AAAInnerTortuositySmallScale']), #18
                   np.average(featureDict['AAAOuterTortuosityLargeScale']), #19
                   np.average(featureDict['AAAOuterTortuositySmallScale']), #20
                   np.max(featureDict['AAAInnerTortuosityLargeScale']), #21
                   np.max(featureDict['AAAInnerTortuositySmallScale']), #22
                   np.max(featureDict['AAAOuterTortuosityLargeScale']), #23
                   np.max(featureDict['AAAOuterTortuositySmallScale']) #24
                   ]
            dataFrame.append(row)
        else:
            print('Couldnt find patient ' + key + ' you should find where this has gone')
    dataFrame = np.array(dataFrame)

    XTransformer = StandardScaler()
    YTransformer = StandardScaler()
    XDataFrame = XTransformer.fit_transform(dataFrame[:, 1:])
    YDataFrame = YTransformer.fit_transform(dataFrame[:, 0].reshape(-1, 1))[:, 0]
    YDataFrame = dataFrame[:, 0]
    listOfData = [{'XTest': XDataFrame[i, :], 'YTest': YDataFrame[i], 'XTrain': np.concatenate([XDataFrame[0:i, :], XDataFrame[i+1:, :]]), 'YTrain': np.concatenate([YDataFrame[0:i], YDataFrame[i+1:]])} for i in range(dataFrame.shape[0]-1)]
    listOfData.append({'XTest': XDataFrame[-1, :], 'YTest': YDataFrame[-1], 'XTrain': XDataFrame[0:-1, :], 'YTrain': YDataFrame[0:-1]})
    return listOfData, YTransformer, dataFrame[:, 0].reshape(-1, 1)

def main():
    data, transformer, trueYDataFrame = turnFeatureDictsToDataFrame()
    #data = createData()

    #randomForestRegression(data, transformer)
    #print()
    #print()
    gaussianRegression(data, transformer, trueYDataFrame)
    print()
    print()
    svmRegression(data, transformer)
    print()
    print()
    mlpRegressor(data, transformer)

if __name__ == '__main__':
    main()
