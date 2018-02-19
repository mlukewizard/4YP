import os
import dicom
from shutil import copyfile

def extractDicomsFromFolder(folderPath, patientCode, writeDir = 'NaN'):
    if writeDir == 'NaN':
        generateDicomFolder = folderPath + '../' + patientCode + '_' + folderPath.split('/')[-2] + '_rearranged/'
    else:
        generateDicomFolder = writeDir + patientCode + '_' + folderPath.split('/')[-2] + '_rearranged/'
    if not os.path.exists(generateDicomFolder):
            os.mkdir(generateDicomFolder)

    fileList = sorted(os.listdir(folderPath))
    fileList = [x for x in fileList if 'dcm' in x]
    imageList = []
    allowableVals = ['AORTA', 'Axial Abdo', '2mm', 'ANGIO', '2.5mm', '25mm', 'Aorta  1.0  B30f']
    if patientCode == 'DG' or patientCode == 'DU' or patientCode == 'EC' or patientCode == 'EM':
        allowableVals = ['AORTA']
    if patientCode == 'DI':
        allowableVals = ['Axial Abdo']
    if patientCode == 'DS' or patientCode == 'EE':
        allowableVals = ['AXIAL 2mm ANGIO']
    if patientCode == 'DY':
        allowableVals = ['ANGIO']
    if patientCode == 'EY':
        allowableVals = ['AbdomenPelvis']
    if patientCode == 'GK':
        ['Axial Abdo', 'Axial Chest']
    if patientCode == 'GU':
        allowableVals = ['ARTERIAL']
    if patientCode == 'HE':
        allowableVals = ['2.5mm']
    if patientCode == 'HK' or patientCode == 'HM':
        allowableVals = ['AXIAL']
    if patientCode == 'HU':
        allowableVals = ['no SP']


    for file in fileList:
        imageList.append(dicom.read_file(folderPath + file))

    imageList = [x for x in imageList if hasattr(x, 'SeriesDescription') and any(myString in x.SeriesDescription for myString in allowableVals)]
    #imageList = [x for x in imageList if hasattr(x, 'SeriesDescription') and x.SeriesDescription == 'Axial Abdo']

    imageList.sort(key= lambda x:x.InstanceNumber)
    for image, i in zip(imageList, range(len(imageList))):
        copyfile(folderPath + image.SOPInstanceUID + '.dcm', generateDicomFolder + 'dicom' + "%03d" % (i,) + '.dcm')

def incrementPatientCode(patientCode):
    firstHalf = patientCode[0]
    secondHalf = patientCode[1]
    if secondHalf == 'Z':
        newSecondHalf = 'A'
    else:
        newSecondHalf = chr(ord(secondHalf) + 1)
    if newSecondHalf == 'A':
        newFirstHalf = chr(ord(firstHalf) + 1)
    else:
        newFirstHalf = firstHalf
    return newFirstHalf + newSecondHalf

def checkAllSortedFolders(motherFolder):
    for subFolder in os.listdir(motherFolder):
        seriesDescriptions= []
        for file in os.listdir(motherFolder + subFolder):
            dicomFile = dicom.read_file(motherFolder + subFolder + '/' + file)
            if hasattr(dicomFile, 'SeriesDescription') and dicomFile.SeriesDescription not in seriesDescriptions:
                seriesDescriptions.append(dicomFile.SeriesDescription)
        if len(seriesDescriptions) > 1:
            printString = 'Folder ' + motherFolder + subFolder + '/' + ' has multiple series descriptions in it, they are '
            for desc in seriesDescriptions:
                printString = printString + desc + ', '
            print(printString)
        else:
            print('Folder ' + motherFolder + subFolder + ' is fine')


def extractAllDicomFolders(motherFilePath, patientCode, rearrangedOutputFolder = 'NaN'):
    filePaths = [motherFilePath]
    if rearrangedOutputFolder == 'NaN':
        writeFilePath = motherFilePath
    else:
        writeFilePath = rearrangedOutputFolder
    while len(filePaths) > 0:
        myList = os.listdir(filePaths[0])
        for entry in myList:
            if '.dcm' in entry:
                if os.path.exists((writeFilePath + patientCode + '_' + filePaths[0].split('/')[-2] + '_rearranged/').replace('/', '\\')):
                    if len(os.listdir((writeFilePath + patientCode + '_' + filePaths[0].split('/')[-2] + '_rearranged/').replace('/', '\\'))) < 1:
                        extractDicomsFromFolder(filePaths[0], patientCode, writeFilePath)
                        print('Saved a dicom folder for filepath ' + filePaths[0] + ' Patient code is ' + patientCode)
                        print('New folder length is ' + str(len(os.listdir((writeFilePath + patientCode + '_' + filePaths[0].split('/')[-2] + '_rearranged/').replace('/', '\\')))))
                    patientCode = incrementPatientCode(patientCode)
                    patientCode = incrementPatientCode(patientCode)
                else:
                    extractDicomsFromFolder(filePaths[0], patientCode, writeFilePath)
                    print('Saved a dicom folder for filepath ' + filePaths[0] + ' Patient code is ' + patientCode)
                    patientCode = incrementPatientCode(patientCode)
                    patientCode = incrementPatientCode(patientCode)
                break
            else:
                filePaths.append(filePaths[0] + entry + '/')
        filePaths.pop(0)

def main():
    extractAllDicomFolders('D:/Tertiary Case Set/', 'CA', 'D:/Tertiary Case Set Rearranged/')
    #checkAllSortedFolders('D:/Tertiary Case Set Rearranged/')

if __name__ == '__main__':
    main()