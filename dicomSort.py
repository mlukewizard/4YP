import os
import dicom
from shutil import copyfile

def extractDicomsFromFolder(folderPath, writeDir = 'NaN'):
    if writeDir == 'NaN':
        generateDicomFolder = folderPath + '../' + folderPath.split('/')[-2] + '_rearrangedDicoms/'
    else:
        generateDicomFolder = writeDir + folderPath.split('/')[-2] + '_rearrangedDicoms/'
    if not os.path.exists(generateDicomFolder):
            os.mkdir(generateDicomFolder)

    fileList = sorted(os.listdir(folderPath))
    fileList = [x for x in fileList if 'dcm' in x]
    imageList = []

    for file in fileList:
        imageList.append(dicom.read_file(folderPath + file))

    imageList = [x for x in imageList if hasattr(x, 'SeriesDescription') and '25mm' in x.SeriesDescription]
    imageList.sort(key= lambda x:x.InstanceNumber)
    for image, i in zip(imageList, range(len(imageList))):
        copyfile(folderPath + image.SOPInstanceUID + '.dcm', generateDicomFolder + 'dicom' + "%03d" % (i,) + '.dcm')

def extractAllDicomFolders(motherFilePath):
    filePaths = [motherFilePath]
    while len(filePaths) > 0:
        myList = os.listdir(filePaths[0])
        for entry in myList:
            if '.dcm' in entry:
                extractDicomsFromFolder(filePaths[0], motherFilePath)
                print('Saved a dicom folder for filepath ' + filePaths[0])
                break
            else:
                filePaths.append(filePaths[0] + entry + '/')
        filePaths.pop(0)

def main():
    extractAllDicomFolders('C:/Users/Luke/Documents/sharedFolder/4YP/segmentations/Secondary Case Set/')

if __name__ == '__main__':
    main()