import os
import dicom
from shutil import copyfile

folderPath = 'C:\\Users\\Luke\\Downloads\\Download+TP3+cases\\S1-14 TP3 processed\\S1-14 TP3 rearranged\\'
generateDicomFolder = 'C:\\Users\\Luke\\Downloads\\Download+TP3+cases\\rearrangedDicoms\\'
if not os.path.exists(generateDicomFolder):
        os.mkdir(generateDicomFolder)

fileList = sorted(os.listdir(folderPath))
fileList = [x for x in fileList if 'dcm' in x]
imageList = []

for file in fileList:
    imageList.append(dicom.read_file(folderPath + file))

imageList = [x for x in imageList if hasattr(x, 'SeriesDescription') and '25mm' in x.SeriesDescription]
imageList.sort(key= lambda x:x.InstanceNumber)
for image, i in zip(imageList, range(len(imageList))): copyfile(folderPath + image.SOPInstanceUID + '.dcm', generateDicomFolder + 'dicom' + "%03d" % (i,) + '.dcm')
