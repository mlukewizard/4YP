import os
import dicom
import numpy as np
import sys

def main():
    #extractPatientDictionaries()
    getContrastInfo()

def getContrastInfo():
    dicomFoldersDir = 'D:/myRearrangedFolders/'
    dicomFolderList = sorted(os.listdir(dicomFoldersDir))
    usefulFiles = {}
    for i, subFolder in enumerate(dicomFolderList):
        dicomFiles = sorted(os.listdir(dicomFoldersDir + subFolder + '/'))
        fileAttributes = dicom.read_file(dicomFoldersDir + subFolder + '/' + dicomFiles[10])
        doubleCheckFileAttributes = dicom.read_file(dicomFoldersDir + subFolder + '/' + dicomFiles[15])
        #if subFolder == "RY_S1-14S1-14 TP3_rearranged":
        #    print('hi')
        patientName = subFolder.split('_')[1].split('T')[0]

        if hasattr(fileAttributes, 'ContrastBolusAgent'):
            if patientName in list(usefulFiles.keys()):
                usefulFiles[patientName] = usefulFiles[patientName] + 1
            else:
                usefulFiles[patientName] = 1
            print('YES agent for ' + subFolder)
        else:
            print('NO agent for ' + subFolder)
    print('hi')

def extractMachineNames():
    dicomFoldersDir = 'D:/newCases/newCasesWithMultipleScansOnGoodMachines/'
    dicomFolderList = sorted(os.listdir(dicomFoldersDir))
    for i, folder in enumerate(dicomFolderList):
        dicomFolderList = sorted(os.listdir(dicomFoldersDir + folder + '/'))
        usefulFolderCount = 0
        gotMinusTwoFolder = False

        for j, subFolder in enumerate(dicomFolderList):
            if 'TP3' in subFolder:
                usefulFolderCount = usefulFolderCount + 1
                dicomFiles = sorted(os.listdir(dicomFoldersDir + folder + '/' + subFolder + '/'))
                fileAttributes = dicom.read_file(dicomFoldersDir + folder + '/' + subFolder + '/' + dicomFiles[10])
                doubleCheckFileAttributes = dicom.read_file(dicomFoldersDir + folder + '/' + subFolder + '/' + dicomFiles[15])
                if fileAttributes.ManufacturerModelName != doubleCheckFileAttributes.ManufacturerModelName:
                    print('Inconsistant machine for tp3: One is ' + fileAttributes.ManufacturerModelName + ' and the other is ' + doubleCheckFileAttributes.ManufacturerModelName)
                print('For folder ' + str(folder) + ' tp3 machine is ' + str(fileAttributes.ManufacturerModelName))
                if hasattr(fileAttributes, 'ContentDate'):
                    if fileAttributes.ContentDate != doubleCheckFileAttributes.ContentDate:
                        print('Inconsistant date for  tp3: One is ' + fileAttributes.ContentDate + ' and the other is ' + doubleCheckFileAttributes.ContentDate)
                    print('For folder ' + str(folder) + ' tp3 date is ' + str(fileAttributes.ContentDate))
                else:
                    print('No ContentDate for tp3')

            if 's 1' in subFolder or 's  1' in subFolder:
                usefulFolderCount = usefulFolderCount + 1
                dicomFiles = sorted(os.listdir(dicomFoldersDir + folder + '/' + subFolder + '/'))
                fileAttributes = dicom.read_file(dicomFoldersDir + folder + '/' + subFolder + '/' + dicomFiles[10])
                doubleCheckFileAttributes = dicom.read_file(dicomFoldersDir + folder + '/' + subFolder + '/' + dicomFiles[15])
                if fileAttributes.ManufacturerModelName != doubleCheckFileAttributes.ManufacturerModelName:
                    print('Inconsistant machine for minus 1: One is ' + fileAttributes.ManufacturerModelName + ' and the other is ' + doubleCheckFileAttributes.ManufacturerModelName)
                print('For folder ' + str(folder) + ' minus 1 machine is ' + str(fileAttributes.ManufacturerModelName))
                if hasattr(fileAttributes, 'ContentDate'):
                    if fileAttributes.ContentDate != doubleCheckFileAttributes.ContentDate:
                        print('Inconsistant date for  minus 1: One is ' + fileAttributes.ContentDate + ' and the other is ' + doubleCheckFileAttributes.ContentDate)
                    print('For folder ' + str(folder) + ' minus 1 date is ' + str(fileAttributes.ContentDate))
                else:
                    print('No ContentDate for minus 1')

            if 's 2' in subFolder or 's  1' in subFolder:
                usefulFolderCount = usefulFolderCount + 1
                dicomFiles = sorted(os.listdir(dicomFoldersDir + folder + '/' + subFolder + '/'))
                fileAttributes = dicom.read_file(dicomFoldersDir + folder + '/' + subFolder + '/' + dicomFiles[10])
                doubleCheckFileAttributes = dicom.read_file(dicomFoldersDir + folder + '/' + subFolder + '/' + dicomFiles[15])
                if fileAttributes.ManufacturerModelName != doubleCheckFileAttributes.ManufacturerModelName:
                    print('Inconsistant machine for minus 2: One is ' + fileAttributes.ManufacturerModelName + ' and the other is ' + doubleCheckFileAttributes.ManufacturerModelName)
                print('For folder ' + str(folder) + ' minus 2 machine is ' + str(fileAttributes.ManufacturerModelName))
                if hasattr(fileAttributes, 'ContentDate'):
                    if fileAttributes.ContentDate != doubleCheckFileAttributes.ContentDate:
                        print('Inconsistant date for  minus 2: One is ' + fileAttributes.ContentDate + ' and the other is ' + doubleCheckFileAttributes.ContentDate)
                    print('For folder ' + str(folder) + ' minus 2 date is ' + str(fileAttributes.ContentDate))
                else:
                    print('No ContentDate for minus 2')

        if usefulFolderCount < 2:
            print('We didnt get 2 folders for ' + folder)
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')


def extractPatientDictionaries():
    dicomFoldersDir = 'D:/allCases/'
    dicomFolderList = sorted(os.listdir(dicomFoldersDir))
    dictOfThicknesses = {}
    dictOfSpacings = {}
    for i, folder in enumerate(dicomFolderList):
        patientID = folder[0:2]
        dicomFiles = sorted(os.listdir(dicomFoldersDir + folder))
        fileAttributes = dicom.read_file(dicomFoldersDir + folder + '/' + dicomFiles[3])
        doubleCheckFileAttributes = dicom.read_file(dicomFoldersDir + folder + '/' + dicomFiles[10])
        if fileAttributes.SliceThickness != doubleCheckFileAttributes.SliceThickness:
            print('This is dodgy')
        if fileAttributes.PixelSpacing != doubleCheckFileAttributes.PixelSpacing:
            print('This is dodgy')
        dictOfThicknesses[patientID] = fileAttributes.SliceThickness
        dictOfSpacings[patientID] = fileAttributes.PixelSpacing
        print(patientID + ' = ' + fileAttributes.ManufacturerModelName)
    # np.save('../Spacings.npy', dictOfSpacings)
    # np.save('../Thicknesses.npy', dictOfThicknesses)

if __name__ == '__main__':
    main()