import os
import dicom
from shutil import copyfile

def moveEmptyFolders(folder1, folder2):
    for folder in sorted(os.listdir(folder1)):
        files = sorted(os.listdir(folder1 + folder + '/'))
        if len(files) < 1:
            print('Moving folder ' + folder)
            os.rmdir(folder1 + folder + '/')



def extractDicomsFromFolder(folderPath, patientCode, writeDir = 'NaN'):
    if writeDir == 'NaN':
        generateDicomFolder = folderPath + '../' + patientCode + '_' + folderPath.split('/')[-2] + '_rearranged/'
    else:
        if 'Machines' in folderPath.split('/')[-3]:
            generateDicomFolder = writeDir + patientCode + '_' + folderPath.split('/')[-2] + '_rearranged/'
        else:
            generateDicomFolder = writeDir + patientCode + '_' + folderPath.split('/')[-3] + folderPath.split('/')[-2] + '_rearranged/'
    if not os.path.exists(generateDicomFolder):
            os.mkdir(generateDicomFolder)

    fileList = sorted(os.listdir(folderPath))
    fileList = [x for x in fileList if 'dcm' in x]
    imageList = []

    for file in fileList:
        imageList.append(dicom.read_file(folderPath + file))
    print('Machine is ' + imageList[10].ManufacturerModelName)

    allowableVals = ['AORTA', 'ANGIO', 'Aorta', '1.25mm Axi', '1.25mm axi', '1.25mm', '1 25mm', '2.5MM', '2.5mm', '1,25mm pos', '30f', 'Axial Abdo']
    allowableVals = ['1.25mm post', '30f']
    disallowableVals = ['Lung', 'lung', 'Pre contrast', 'Pre Con', 'Bone', 'bone' 'COR', 'Coronial', 'SAG', 'Sagatial', 'sagatial', 'coronial', 'Recon 2']

    imageList = [x for x in imageList if hasattr(x, 'SeriesDescription') and any(myString in x.SeriesDescription for myString in allowableVals) and
                 all(myString not in x.SeriesDescription for myString in disallowableVals)]

    imageList.sort(key= lambda x:x.InstanceNumber)
    for image, i in zip(imageList, range(len(imageList))):
        copyfile(folderPath + image.SOPInstanceUID + '.dcm', generateDicomFolder + 'dicom' + "%03d" % (i,) + '.dcm')


    checkSortedFolder(generateDicomFolder)

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

def checkSortedFolder(subFolder):
    seriesDescriptions = []
    machines = []
    if len(os.listdir(subFolder)) < 1:
        print('Folder ' + subFolder + ' is empty')
    for file in os.listdir(subFolder):
        dicomFile = dicom.read_file(subFolder + '/' + file)
        if hasattr(dicomFile, 'SeriesDescription') and dicomFile.SeriesDescription not in seriesDescriptions:
            seriesDescriptions.append(dicomFile.SeriesDescription)
        if hasattr(dicomFile, 'ManufacturerModelName') and dicomFile.ManufacturerModelName not in machines:
            machines.append(dicomFile.ManufacturerModelName)
    if len(seriesDescriptions) > 1:
        printString = 'Folder ' + subFolder + '/' + ' has multiple series descriptions in it, they are '
        for desc in seriesDescriptions:
            printString = printString + desc + ', '
        print(printString)
    else:
        if len(os.listdir(subFolder)) > 0:
            print('Folder ' + subFolder + ' is consistant with seriesDescription ' + seriesDescriptions[0])
    if len(machines) > 1:
        print('Warning there are two registered machines ' + machines[0] + ' and ' + machines[1])
    elif len(machines) == 1:
        print('Machine is ' + machines[0])

def checkAllSortedFolders(motherFolder):
    for subFolder in os.listdir(motherFolder):
        checkSortedFolder(motherFolder + subFolder + '/')


def extractAllDicomFolders(motherFilePath, patientCode, rearrangedOutputFolder = 'NaN'):
    filePaths = [motherFilePath]
    if rearrangedOutputFolder == 'NaN':
        writeFilePath = motherFilePath
    else:
        writeFilePath = rearrangedOutputFolder
    alreadyProcessedFolders = sorted(os.listdir(writeFilePath))
    while len(filePaths) > 0:
        myList = os.listdir(filePaths[0])
        for entry in myList:
            if '.dcm' in entry:
                if any([True if patientCode in folder[0:2] else False for folder in alreadyProcessedFolders]):
                    print('Already done patient ' + patientCode)
                    #for folder in alreadyProcessedFolders:
                    #    if patientCode in folder[0:2]:
                    #        checkSortedFolder(writeFilePath + folder + '/')
                    patientCode = incrementPatientCode(patientCode)
                    patientCode = incrementPatientCode(patientCode)
                else:
                    print('Extracting patient ' + patientCode)
                    extractDicomsFromFolder(filePaths[0], patientCode, writeFilePath)
                    print('Saved a dicom folder for filepath ' + filePaths[0] + ' Patient code is ' + patientCode)
                    patientCode = incrementPatientCode(patientCode)
                    patientCode = incrementPatientCode(patientCode)
                print('~~~~~~~~~~~~~~~~~~~~~~~')
                break
            else:
                filePaths.append(filePaths[0] + entry + '/')
        filePaths.pop(0)

def main():
    #extractAllDicomFolders('D:/newCases/newCasesWithMultipleScansOnGoodMachines/', 'IA', 'D:/myRearrangedFolders/')
    moveEmptyFolders('D:/myRearrangedFolders/', 'D:/myRearrangedFolders/')
    #checkAllSortedFolders('D:/allCases/')
    #extractDicomsFromFolder('D:/Tertiary Case Set/S2-52 TP3/', 'ZX')

if __name__ == '__main__':
    main()