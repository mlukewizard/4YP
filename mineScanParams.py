import os
import dicom
import numpy as np

def main():
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
    np.save('../Spacings.npy', dictOfSpacings)
    np.save('../Thicknesses.npy', dictOfThicknesses)

if __name__ == '__main__':
    main()