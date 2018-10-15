import warnings
try:
    import pydicom as dicom
except ImportError:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import dicom
from tkinter import *
from tkinter import filedialog
import tkinter as tk
import tkinter.ttk as ttk
import os
import numpy as np
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import messagebox
from tkinter import simpledialog

lstFilesDCM = []      

def loadFiles():
    resetSelector()
    folder =  filedialog.askdirectory()    
    lstFilenames = []
    for dirName, subdirList, fileList in os.walk(folder):
        for filename in fileList:
            if ".dcm" in filename.lower():
                lstFilesDCM.append(os.path.join(dirName,filename))
                lstFilenames.append(filename)
    
    file_cb["values"] = lstFilenames    
    files_fr.pack() 

def resetSelector():
    global lstFilesDCM
    lstFilesDCM.clear()
    info_text.delete('1.0', END)
    file_cb.set("Select a DICOM file")
    functions_cb.set("Select function")
    apply_bt.configure(state=DISABLED)    
    
def showHeaderInfo(header):
    info_text.delete('1.0', END)
    info_text.config(state=NORMAL)  
    info = "\nPatient ID: "+header.PatientID
    info += "\nManufacturer: "+header.Manufacturer
    info += "\nStudy Description: "+header.StudyDescription
    info += "\nMR Acquisition Type: "+header.MRAcquisitionType
    info += "\nSpacing Between Slices: "+str(header.SpacingBetweenSlices)
    info += "\nPixel Bandwidth: "+str(header.PixelBandwidth)
    info += "\nRows: "+str(header.Rows)
    info += "\nColumns: "+str(header.Columns)
    info += "\nPixel Spacing : "+str(header.PixelSpacing)
    info_text.delete('1.0', END)
    info_text.insert('1.0', info)
    info_text.config(state=DISABLED)
    
def hist():    
    imageInfo = dicom.read_file(lstFilesDCM[file_cb.current()])
    rows = int(imageInfo.Rows)
    columns = int(imageInfo.Columns)
    pixelArray = imageInfo.pixel_array    
    intensity = [0]*65536
    
    for i in range(rows):
        for j in range(columns):
            intensity[pixelArray[i,j]]=intensity[pixelArray[i,j]]+1
            
    intensity = np.asarray(intensity)
    plt.plot(intensity)

    fig = plt.gcf()
    fig.canvas.set_window_title('Histogram')       
    
    plt.show()       
            
def processImage():
    global canvas        
    text_fr.pack()   
    apply_bt.configure(state=NORMAL)

    imageInfo = dicom.read_file(lstFilesDCM[file_cb.current()])
    showHeaderInfo(imageInfo)
    
    plt.set_cmap(plt.gray())
    f = Figure()
    a = f.add_subplot(111)    
    a.imshow(np.flipud(imageInfo.pixel_array))     
    canvas.get_tk_widget().destroy()
    canvas = FigureCanvasTkAgg(f, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack()    

def getBaseMatrix(borderType, borderSize):    
    image = dicom.read_file(lstFilesDCM[file_cb.current()]) 
    if borderType == 1:
        return np.pad(image.pixel_array, borderSize, mode='symmetric')
    elif borderType == 2:
        return np.pad(image.pixel_array, borderSize, mode='edge')
    elif borderType == 3:
        return image.pixel_array
    

def getValueFromProduct(matrix, kernel):
    size = np.shape(matrix)[0]
    result = 0

    for i in range(0, size):
        for j in range(0, size):
            result += matrix[i,j] * kernel[i,j]

    return int(result)

def applyConvolution(matrix, kernel, borderSize, borderType):
    rowsLimit = np.shape(matrix)[0] - borderSize
    columnsLimit = np.shape(matrix)[1] - borderSize
    convMatrix = matrix

    for i in range(borderSize, rowsLimit):
        for j in range(borderSize, columnsLimit):
            submatrix = matrix[i-1:i+2:1,j-1:j+2:1]
            convMatrix[i,j] = getValueFromProduct(submatrix, kernel)
    
    if borderType == 3:
        finalMatrix = convMatrix
    else:
        finalMatrix = convMatrix[borderSize:rowsLimit:1, borderSize:columnsLimit:1]   

    plt.imshow(np.flipud(finalMatrix))
    plt.gcf().canvas.set_window_title('Low Pass Filter')     
    plt.show()
     
def averageFilter(kernelSize, borderType):        
    kernel = np.ones((kernelSize, kernelSize))
    kernelFactor = 1/np.sum(kernel)    
    borderSize = int((kernelSize-1)/2)
    matrix = getBaseMatrix(borderType, borderSize)

    applyConvolution(matrix, kernel*kernelFactor, borderSize, borderType)

    
def applyFunction():
    function = functions_cb.get()    
    if function == 'Histogram':
        hist()
    elif function == 'Average Filter':        
        kernelSize = simpledialog.askinteger("Kernel Size", "Digit the kernel size\n", parent=root, minvalue=3)       
        if kernelSize % 2 == 0:
            messagebox.showinfo("Warning","The kernel size must be an odd number")
            return
        
        borderType = simpledialog.askinteger("Border Type", "Digit the border type\n\n1. Mirror\n\n2. Replicate\n\n3. Ignore\n", parent=root, minvalue=1, maxvalue=3)
        averageFilter(kernelSize, borderType)
    else:
        messagebox.showinfo("Error", "Function not found")    

#TK components
root = tk.Tk()
root.title("Medical Imaging")
root.configure(background='pink')
root.geometry('%dx%d+%d+%d' % (700, root.winfo_screenheight(), 0, 0))

selectFolder_fr = Frame(root)
selectFolder_fr.configure(background='pink')
selectFolder_fr.pack()

selectFolder_bt = tk.Button(selectFolder_fr, text="Select folder", command=loadFiles, bg='white')
selectFolder_bt.pack(pady=5)

files_fr =Frame(root)
files_fr.configure(background='pink')

file_cb = ttk.Combobox(files_fr, state='readonly')
file_cb.set("Select a DICOM file")
file_cb.grid(row=0, column=0)

process_bt = tk.Button(files_fr, text="Process", command=processImage, bg='white')
process_bt.grid(row=0, column=1, padx=5)

functions_cb = ttk.Combobox(files_fr, state='readonly')
functions_cb.set("Select function")
functions_cb.grid(row=1, column=0, pady=10)
functions_cb["values"] = ['Histogram', 'Average Filter']

apply_bt = tk.Button(files_fr, text="Apply", command=applyFunction, bg='white', state=DISABLED)
apply_bt.grid(row=1, column=1, pady=10, padx=5)

text_fr = Frame(root)
text_fr.configure(background='pink')

info_text = tk.Text(text_fr, width = 90, height = 11)
info_text.pack(pady=20)

f = Figure()
canvas = FigureCanvasTkAgg(f, master=root)

root.mainloop()