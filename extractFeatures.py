import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import csv
import random

patientID = 'NS'
pointCloudFiles = ['C:\\Users\\Luke\\Documents\\sharedFolder\\4YP\\pointClouds\\Regent_' + patientID + '\\' + patientID + 'InnerPointCloud.csv', 'C:\\Users\\Luke\\Documents\\sharedFolder\\4YP\\pointClouds\\Regent_' + patientID + '\\' + patientID + 'OuterPointCloud.csv']

pointList = []
pointClouds = []

for pointCloudFile in pointCloudFiles:
    with open(pointCloudFile, 'rb') as csvfile:
        fileReader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        next(fileReader)
        for row in fileReader:
            if random.uniform(0, 1) > 0.9:
                row = row[0].split(',')
                row = [int(row[0]), int(row[1]), int(row[2])]
                pointList.append(row)
        pointClouds.append(np.array(pointList))

plt.figure()
ax = plt.subplot(projection='3d')
ax.scatter(pointClouds[1][:, 0], pointClouds[1][:, 1], -pointClouds[1][:, 2], zdir='z', c='red', marker='.')
plt.show()

