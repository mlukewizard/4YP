import numpy as np
import os
import sys
import csv
from random import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.svm import SVR
from sklearn.datasets import make_regression

def randomForestRegression(data):
    regr =RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
           max_features='auto', max_leaf_nodes=None,
           min_impurity_decrease=0.0, min_impurity_split=None,
           min_samples_leaf=1, min_samples_split=2,
           min_weight_fraction_leaf=0.0, n_estimators=1000, n_jobs=1,
           oob_score=False, random_state=0, verbose=0, warm_start=False)
    regr.fit(data['XTrain'], data['YTrain'])
    print('Predicting using random forest')
    error = 0
    for i, answer in enumerate(data['YTest']):
        print('Predicted: ' + str(regr.predict([data['XTest'][i, :]])) + ' Label: ' + str(answer))
        error += abs(answer - regr.predict([data['XTest'][i, :]]))
    print('Total error is ' + str(error))


def svmRegression(data):
    segr = SVR(degree=3)

    segr.fit(data['XTrain'], data['YTrain'])

    print('Predicting using svm regressor')
    error = 0
    for i, answer in enumerate(data['YTest']):
        print('Predicted: ' + str(segr.predict([data['XTest'][i, :]])) + ' Label: ' + str(answer))
        error += abs(answer - segr.predict([data['XTest'][i, :]]))
    print('Total error is ' + str(error))



def gaussianRegression(data):
    Gregr = GaussianProcessRegressor(kernel=None, alpha=1e-10, n_restarts_optimizer=0, normalize_y=True, copy_X_train=True, random_state=None)

    Gregr.fit(data['XTrain'], data['YTrain'])

    print('Predicting using gaussian regressor')
    error = 0
    for i, answer in enumerate(data['YTest']):
        print('Predicted: ' + str(Gregr.predict([data['XTest'][i, :]])) + ' Label: ' + str(answer))
        error += abs(answer - Gregr.predict([data['XTest'][i, :]]))
    print('Total error is ' + str(error))


def createData():
    randomData = np.random.rand(1000, 10)
    answers = np.random.rand(randomData.shape[0])
    for i in range(randomData.shape[0]):
        row = randomData[i, :]
        answers[i] = row[0] + 0.118 * row[1] - 0.372 * row[2] + 0.742 * row[3] + 0.646 * row[4] - 0.846 * np.square(row[5]) + 0.238 * np.sin(6 * row[6])
    XTrain = randomData[0:-30, :]
    XTest = randomData[-30:, :]
    YTrain = answers[0:-30]
    YTest = answers[-30:]
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
            row = [FMDVal, featureDict['maxInnerArea'], featureDict['outerAorticVolume']]
            dataFrame.append(row)
        else:
            print('Couldnt find patient ' + key + ' you should find where this has gone')
    dataFrame = np.array(dataFrame)
    return {'XTest': dataFrame[-5:, 1:], 'YTest': dataFrame[-5:, 0:1], 'XTrain': dataFrame[0:-5, 1:], 'YTrain': dataFrame[0:-5:, 0:1]}

def main():
    data = turnFeatureDictsToDataFrame()
    randomForestRegression(data)
    print()
    print()
    gaussianRegression(data)
    print()
    print()
    svmRegression(data)

if __name__ == '__main__':
    main()
