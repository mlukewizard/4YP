import matplotlib
matplotlib.use('TkAgg')
from numpy import arange, sin, pi
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
# implement the default mpl key bindings
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
import sys
if sys.version_info[0] < 3:
    import Tkinter as Tk
else:
    import tkinter as Tk
import os
import numpy as np
import dicom
import sys
import matplotlib.pyplot as plt

patientID = 'CS'
dicomDir = 'D:/allCases/' + [file for file in os.listdir('D:/allCases/') if file[0:2] == patientID][0] + '/'
InnerPC = np.load('D:/processedCases/' + patientID + '_processed/' + patientID + 'ThickInnerPointCloud.npy')
OuterPC = np.load('D:/processedCases/' + patientID + '_processed/' + patientID + 'ThickOuterPointCloud.npy')

'''
filePath = '../'
localFiles = os.listdir(filePath)
for file in localFiles:
    if 'nner' in file:
        InnerPC = np.load(filePath + file)
    elif 'uter' in file:
        OuterPC = np.load(filePath + file)
    elif 'dicoms' in file:
        dicomDir = filePath + file + '/'
'''

if InnerPC.shape[0] != len(os.listdir(dicomDir)) or OuterPC.shape[0] != InnerPC.shape[0]:
    sys.exit('Your dimensions arent equal')

root = Tk.Tk()
root.wm_title("Segmentation viewer")

f = plt.figure(figsize=(5,4))
dicomImage = dicom.read_file(dicomDir + sorted(os.listdir(dicomDir))[1]).pixel_array
plt.imshow(dicomImage, cmap='gray')
plt.axis('off')

# a tk.DrawingArea
canvas = FigureCanvasTkAgg(f, master=root)
canvas.show()
canvas.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)


def quitProgram():
    root.quit()     # stops mainloop
    root.destroy()  # this is necessary on Windows to prevent

def updateSlice():
    outerContourOn = False
    innerContourOn = False
    sliceNumber = int(sliceNumberBox.get())
    dicomImage = dicom.read_file(dicomDir + sorted(os.listdir(dicomDir))[sliceNumber]).pixel_array
    #misc.imsave(tmpFolder + 'dicom.png', dicomImage)
    plt.imshow(dicomImage, cmap='gray')
    plt.axis('off')
    canvas.draw()

def changeInnerContour():
    sliceNumber = int(sliceNumberBox.get())
    dicomImage = dicom.read_file(dicomDir + sorted(os.listdir(dicomDir))[sliceNumber]).pixel_array
    innerBinaryOverlay = InnerPC[sliceNumber, :, :]
    innerBinaryOverlay = innerBinaryOverlay[0:-2, 0:-2] + innerBinaryOverlay[2:, 2:] + innerBinaryOverlay[0:-2, 2:] + innerBinaryOverlay[2:, 0:-2]
    innerBinaryOverlay = np.pad(innerBinaryOverlay, 1, 'edge')
    difference = np.where(innerBinaryOverlay > 0, 255, 0) - np.where(InnerPC[sliceNumber, :, :] > 0, 255, 0)
    plt.imshow(np.clip(dicomImage + np.max(dicomImage)*difference, 0, np.max(dicomImage)), cmap='gray')
    plt.axis('off')
    canvas.draw()

def changeOuterContour():
    sliceNumber = int(sliceNumberBox.get())
    dicomImage = dicom.read_file(dicomDir + sorted(os.listdir(dicomDir))[sliceNumber]).pixel_array
    outerBinaryOverlay = OuterPC[sliceNumber, :, :]
    outerBinaryOverlay = outerBinaryOverlay[0:-2, 0:-2] + outerBinaryOverlay[2:, 2:] + outerBinaryOverlay[0:-2, 2:] + outerBinaryOverlay[2:, 0:-2]
    outerBinaryOverlay = np.pad(outerBinaryOverlay, 1, 'edge')
    difference = np.where(outerBinaryOverlay > 0, 255, 0) - np.where(OuterPC[sliceNumber, :, :] > 0, 255, 0)
    plt.imshow(np.clip(dicomImage + np.max(dicomImage) * difference, 0, np.max(dicomImage)), cmap='gray')
    plt.axis('off')
    canvas.draw()

button1 = Tk.Button(master=root, text='Quit', command=quitProgram)
button2 = Tk.Button(master=root, text='Update Viewer/Remove Contour', command=updateSlice)
button3 = Tk.Button(master=root, text='Show Inner Contour', command=changeInnerContour)
button4 = Tk.Button(master=root, text='Show Outer Contour', command=changeOuterContour)
sliceNumberBox = Tk.Entry(master=root)
sliceNumberBox.insert(Tk.END, '1')
EntryBoxLabel = Tk.Label(master=root, text = 'Slice Number (Max=' + str(int(len(os.listdir(dicomDir)))) + '):')
EntryBoxLabel.pack(side=Tk.LEFT)
sliceNumberBox.pack(side=Tk.LEFT)
button1.pack(side=Tk.BOTTOM)
button3.pack(side=Tk.BOTTOM)
button4.pack(side=Tk.BOTTOM)
button2.pack(side=Tk.BOTTOM)

Tk.mainloop()