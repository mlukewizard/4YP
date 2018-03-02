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
           max_features=9, max_leaf_nodes=None,
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

def main():
    data = createData()
    randomForestRegression(data)
    print()
    print()
    gaussianRegression(data)
    print()
    print()
    svmRegression(data)

if __name__ == '__main__':
    main()