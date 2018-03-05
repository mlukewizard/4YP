import os
import dicom
import numpy as np
from scipy import misc

def main():
    processedCasesDir = 'D:/processedCases/'
    processedCasesList = sorted(os.listdir(processedCasesDir))
    processedCasesList = [dir for dir in processedCasesList if dir[0:2] in ['AE']]
    for i, folder in enumerate(processedCasesList):
        if not any('PointCloud' in file for file in os.listdir(processedCasesDir + folder)) and any('rediction' in file for file in os.listdir(processedCasesDir + folder)):
            patientID = folder[0:2]
            print(patientID)
            innerPredFolder = [file for file in os.listdir(processedCasesDir + folder) if 'innerPrediction' in file][0]
            innerPredList = [misc.imread(processedCasesDir + folder + '/' + innerPredFolder + '/' + file) for file in sorted(os.listdir(processedCasesDir + folder + '/' + innerPredFolder))]
            thickInnerPointCloud = np.array(innerPredList)
            np.save(processedCasesDir + folder + '/' + patientID + 'ThickInnerPointCloud.npy', thickInnerPointCloud)
            del(innerPredList)

            outerPredFolder = [file for file in os.listdir(processedCasesDir + folder) if 'outerPrediction' in file][0]
            outerPredList = [misc.imread(processedCasesDir + folder + '/' + outerPredFolder + '/' + file) for file in sorted(os.listdir(processedCasesDir + folder + '/' + outerPredFolder))]
            thickOuterPointCloud = np.array(outerPredList)
            np.save(processedCasesDir + folder + '/' + patientID + 'ThickOuterPointCloud.npy', thickOuterPointCloud)
            del (outerPredList)
if __name__ == '__main__':
    main()