import numpy as np
import os
import sys
import csv
from random import *
import copy
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import random
from numpy.linalg import norm
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA

regr =RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
           max_features='auto', max_leaf_nodes=None,
           min_impurity_decrease=0.0, min_impurity_split=None,
           min_samples_leaf=1, min_samples_split=2,
           min_weight_fraction_leaf=0.0, n_estimators=1000, n_jobs=1,
           oob_score=False, random_state=0, verbose=0, warm_start=False)

regr = SVR(kernel='rbf', degree=10, gamma='auto', coef0=0.0, tol=0.0000001, C=100000, epsilon=0.0000001, shrinking=False, cache_size=200, verbose=False, max_iter=-1)

regr = MLPRegressor(hidden_layer_sizes=(100, ), activation='relu', solver ='adam', alpha = 0.0001, batch_size ='auto', learning_rate ='constant',
    learning_rate_init = 0.001, power_t = 0.5, max_iter = 200, shuffle = True, random_state = None, tol = 0.0001, verbose = False, warm_start = False,
    momentum = 0.9, nesterovs_momentum = True, early_stopping = False, validation_fraction = 0.1, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-08)

def gaussianRegression(XTrain, YTrain, XTest, YTest):
    myKernel = C(1.0, constant_value_bounds="fixed") * RBF(1.0, length_scale_bounds="fixed")
    regr = GaussianProcessRegressor(kernel=myKernel, alpha=1e-10, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=100, normalize_y=False, copy_X_train=True, random_state=None)
    return testRegressor(regr, XTrain, YTrain, XTest, YTest)

def nDGaussianRegression(XFrame, YFrame):
    myKernel = C(1.0, constant_value_bounds="fixed") * RBF(1.0, length_scale_bounds="fixed")
    regr = GaussianProcessRegressor(kernel=myKernel, alpha=1e-10, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=100, normalize_y=False, copy_X_train=True, random_state=None)
    nDTestRegressor(regr, XFrame, YFrame)

def testRegressor(regr, inputXTrain, inputYTrain, inputXTest, inputYTest):
    XTrain = copy.deepcopy(inputXTrain)
    YTrain = copy.deepcopy(inputYTrain)
    XTest = copy.deepcopy(inputXTest)
    YTest = copy.deepcopy(inputYTest)
    XTrainTransformed, YTrainTransformed, xMultipliers, xMeans, YTransformer = normaliseDataFrames(XTrain, YTrain)
    regr = regr.fit(XTrainTransformed, YTrainTransformed)
    for i in range(XTest.shape[0]):
        XTest[i] = (XTest[i]/xMultipliers[i]) - xMeans[i]

    prediction, bonus = regr.predict(np.expand_dims(XTest, axis=0), return_std=True)
    prediction = YTransformer.inverse_transform(prediction)
    #print('Predicted: ' + str(prediction) + ' Label: ' + str(YTest))
    return prediction, bonus

def nDTestRegressor(regr, XFrame, YFrame):
    print(regr)
    points = []
    error = 0

    for i in range(YFrame.shape[0]):
        XTrain = np.delete(copy.deepcopy(XFrame), i, axis=0)
        YTrain = np.delete(copy.deepcopy(YFrame), i)
        XTest = copy.deepcopy(XFrame)[i, :]
        YTest = copy.deepcopy(YFrame)[i]
        prediction, bonus = testRegressor(regr, XTrain, YTrain, XTest, YTest)
        if bonus < 1:
            points.append([YTest, prediction])
            error += abs(YTest - prediction)
    print('Average absolute error is ' + str(error/YFrame.shape[0]))
    points = np.array(points)

    print('Pearson is ' + str(pearsonr(points[:, 0], points[:, 1])))
    print('Spearman is ' + str(spearmanr(points[:, 0], points[:, 1])))

    p1 = np.array([-3,-3])
    p2 = np.array([5, 5])
    totalError = 0
    for i in range(points.shape[0] - 1, -1, -1):
        p3 = points[i, :]
        totalError = totalError + norm(np.cross(p2 - p1, p1 - p3)) / norm(p2 - p1)
    averageError = totalError / points.shape[0] - 1
    print('Average error is ' + str(averageError))

    plt.scatter(points[:, 0], points[:, 1])
    plt.xlabel('True FMD Value')
    plt.ylabel('Algorithmically predicted FMD Value')
    axes = plt.gca()
    axes.set_xlim([-3, 6])
    axes.set_ylim([-3, 6])
    plt.show()

def gaussianWeightDiscovery(XDataFrame, YDataFrame, averageFeatureVals, averageYVal, featureVars, yVar):
    originalPrediction, bonus = gaussianRegression(XDataFrame, YDataFrame, averageFeatureVals, averageYVal)
    relativeImportances = []
    for featureNum in range(XDataFrame.shape[1]):
        perturbedFeatureVals = copy.deepcopy(averageFeatureVals)
        perturbedFeatureVals[featureNum] = perturbedFeatureVals[featureNum] - featureVars[featureNum]/10
        perturbedPrediction, bonus = gaussianRegression(XDataFrame, YDataFrame, perturbedFeatureVals, averageYVal)
        changeOverVariance = np.abs(perturbedPrediction - originalPrediction)/yVar
        print('Change over variance is ' + str(changeOverVariance))
        relativeImportances.append(changeOverVariance)



def createData():
    featureNum = 20
    randomFunction = np.random.rand(featureNum)-0.4
    XDataFrame = np.random.rand(40, featureNum)
    trueYDataFrame = np.zeros(XDataFrame.shape[0])
    for i in range(XDataFrame.shape[0]):
        trueYDataFrame[i] = np.dot(XDataFrame[i, :], randomFunction) #+ 2*(random.random()-0.9)

    return trueYDataFrame

def normaliseDataFrames(XDataFrame, trueYDataFrame):
    YDataFrame = copy.deepcopy(trueYDataFrame)
    # Scales the data for each feature
    YTransformer = StandardScaler()
    transformedY = YTransformer.fit_transform(YDataFrame.reshape(-1, 1))[:, 0]

    xMultipliers = []
    xMeans = []
    for featureNum in range(int(XDataFrame.shape[1])):
        xMultipliers.append(max(XDataFrame[:, featureNum]))
        XDataFrame[:, featureNum] = XDataFrame[:, featureNum]/max(XDataFrame[:, featureNum])
        xMeans.append(np.mean(XDataFrame[:, featureNum]))
        XDataFrame[:, featureNum] = XDataFrame[:, featureNum] - np.mean(XDataFrame[:, featureNum])

    return XDataFrame, transformedY, xMultipliers, xMeans, YTransformer

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
            XRow =[#np.nanmax(featureDict['innerDiameter'][150:]),  #0
                   #np.nanmax(featureDict['outerDiameter'][150:]),  #1
                   np.nanmax(featureDict['innerDiameter']),
                   np.nanmax(featureDict['outerDiameter']),
                   np.nanmax(featureDict['innerPerimeter']),  #2
                   np.nanmax(featureDict['outerPerimeter']),  #3

                   np.nanmax(featureDict['innerArea']),  #4
                   np.nanmax(featureDict['outerArea']),  #5
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
                   featureDict['bulgeHeight'],  #20

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


    #pca = PCA(n_components=10)
    #XDataFrame = pca.fit_transform(XDataFrame)

    featureVars = []
    averageFeatureVals = []
    for featureNum in range(XDataFrame.shape[1]):
        averageFeatureVals.append(np.mean(XDataFrame[:, featureNum]))
        featureVars.append(np.std(XDataFrame[:, featureNum]))
    averageFMDVal = np.average(trueYDataFrame)
    averageFeatureVals = np.array(averageFeatureVals)
    yVar = np.std(trueYDataFrame)

    #plt.scatter(XDataFrame[:, 5], trueYDataFrame)
    #plt.show()

    return trueYDataFrame, XDataFrame, averageFMDVal, averageFeatureVals, featureVars, yVar

def correlateFeatures(YDataFrame, XDataFrame):
    for featureNum in range(XDataFrame.shape[1]):
        print(np.corrcoef(XDataFrame[:, featureNum], YDataFrame)[0,1])

def main():
    YDataFrame, XDataFrame, averageYVal, averageFeatureVals, featureVars, yVar = turnFeatureDictsToDataFrame()
    #YDataFrame, XDataFrame, averageYVal, averageFeatureVals = createData()

    #gaussianRegression(XDataFrame, YDataFrame, averageFeatureVals, averageYVal)
    #gaussianWeightDiscovery(XDataFrame, YDataFrame, averageFeatureVals, averageYVal, featureVars, yVar)
    nDGaussianRegression(XDataFrame, YDataFrame)
    #correlateFeatures(YDataFrame, XDataFrame)


if __name__ == '__main__':
    main()
