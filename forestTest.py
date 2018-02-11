import numpy as np
import os
import sys
import csv
from random import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.datasets import make_regression

'''
dataFile = 'C:\\Users\\Luke\\Documents\\sharedFolder\\4YP\\RandomForest\\TestData.csv'
trainSize = 1480
testSize = 20
XTrain = np.ndarray([trainSize, 16])
yTrain = np.ndarray([trainSize])
XTest = np.ndarray([testSize, 16])
yTest = np.ndarray([testSize])
with open(dataFile, 'rt') as csvfile:
    fileReader = csv.reader(csvfile, delimiter=',')
    next(fileReader)
    next(fileReader)
    trainIndex = -1
    testIndex = -1
    for row in fileReader:
        trainIndex = trainIndex + 1
        if trainIndex < trainSize:
            XTrain[trainIndex, :] = row[0:-1]
            yTrain[trainIndex] = row[-1]
        else:
            testIndex = testIndex + 1
            XTest[testIndex, :] = row[0:-1]
            yTest[testIndex] = row[-1]
'''

randomData = np.random.rand(1000, 10)
answers = np.random.rand(randomData.shape[0])
for i in range(randomData.shape[0]):
    row = randomData[i, :]
    answers[i] = row[0] + 0.118*row[1] - 0.372*row[2] + 0.742*row[3] + 0.646*row[4] - 0.846*np.square(row[5]) + 0.238*np.sin(6*row[6])
XTrain = randomData[0:-30, :]
XTest = randomData[-30:, :]
YTrain = answers[0:-30]
YTest = answers[-30:]

regr =RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
           max_features=9, max_leaf_nodes=None,
           min_impurity_decrease=0.0, min_impurity_split=None,
           min_samples_leaf=1, min_samples_split=2,
           min_weight_fraction_leaf=0.0, n_estimators=1000, n_jobs=1,
           oob_score=False, random_state=0, verbose=0, warm_start=False)

Gregr = GaussianProcessRegressor(kernel=None, alpha=1e-10, n_restarts_optimizer=0, normalize_y=True, copy_X_train=True, random_state=None)

Gregr.fit(XTrain, YTrain)
regr.fit(XTrain, YTrain)

print('Predicting using random forest')
error = 0
for i, answer in enumerate(YTest):
    print('Predicted: ' + str(regr.predict([XTest[i, :]])) + ' Label: ' + str(answer))
    error += abs(answer - regr.predict([XTest[i, :]]))
print('Total error is ' + str(error))

print()
print()

print('Predicting using gaussian regressor')
error = 0
for i, answer in enumerate(YTest):
    print('Predicted: ' + str(Gregr.predict([XTest[i, :]])) + ' Label: ' + str(answer))
    error += abs(answer - Gregr.predict([XTest[i, :]]))
print('Total error is ' + str(error))