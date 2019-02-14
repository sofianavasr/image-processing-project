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
from kernels import *

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
            
def plotImages(newImage):
    global canvas, canvas2
    originalImage = dicom.read_file(lstFilesDCM[file_cb.current()]).pixel_array

    plt.set_cmap(plt.gray())
    f = Figure(figsize=(90,90))
    a = f.add_subplot(121)  
    a2 = f.add_subplot(122)

    a.imshow(originalImage)
    canvas.get_tk_widget().destroy()
    canvas = FigureCanvasTkAgg(f, master=left_image_fr)
    
    canvas.draw()
    canvas.get_tk_widget().pack()    
   
    a2.imshow(newImage)
    canvas2.get_tk_widget().destroy()
    canvas2 = FigureCanvasTkAgg(f, master=right_image_fr)
    
    canvas2.draw()
    canvas2.get_tk_widget().pack()

def processImage():        
    text_fr.pack()   
    images_fr.pack(fill='both', expand=True)
    left_image_fr.pack(fill='both', expand=True)
    right_image_fr.pack(fill='both', expand=True)
  
    apply_bt.configure(state=NORMAL)

    imageInfo = dicom.read_file(lstFilesDCM[file_cb.current()])
    showHeaderInfo(imageInfo)

    plotImages(imageInfo.pixel_array)
        
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

    return result

def applyConvolution(matrix, kernel, borderSize, borderType):
    shape = np.shape(matrix)
    rowsLimit = shape[0] - borderSize
    columnsLimit = shape[1] - borderSize
    convMatrix = np.copy(matrix)
    
    for i in range(borderSize, rowsLimit):
        for j in range(borderSize, columnsLimit):
            submatrix = matrix[i-1:i+2:1,j-1:j+2:1]           
            convMatrix[i,j] = getValueFromProduct(submatrix, kernel)
    
    if borderType == 3:
        finalMatrix = convMatrix
    else:
        finalMatrix = convMatrix[borderSize:rowsLimit:1, borderSize:columnsLimit:1]   
    
    """ plt.imshow(finalMatrix)
    plt.gcf().canvas.set_window_title('Image Filtering')     
    plt.show()
 """
    plotImages(finalMatrix)
    return finalMatrix
   
def averageFilter(kernelSize, borderType):        
    kernel = np.ones((kernelSize, kernelSize))
    kernelFactor = 1/np.sum(kernel)    
    borderSize = int((kernelSize-1)/2)
    matrix = getBaseMatrix(borderType, borderSize)

    applyConvolution(matrix, kernel*kernelFactor, borderSize, borderType)

def getGaussianKernel(sigma, kernelSize):
    if sigma == 0.5:
        if kernelSize == 3:
            return kernel053
        elif kernelSize == 5:
            return kernel055
        elif kernelSize == 7:
            return kernel057
        else:
            return kernel0511
    elif sigma == 1.0:
        if kernelSize == 3:
            return kernel13
        elif kernelSize == 5:
            return kernel15
        elif kernelSize == 7:
            return kernel17
        else:
            return kernel111
    else:
        if kernelSize == 3:
            return kernel153
        elif kernelSize == 5:
            return kernel155
        elif kernelSize == 7:
            return kernel157
        else:
            return kernel1511

def gaussianFilter(sigma, kernelSize, borderType):
    kernel = getGaussianKernel(sigma, kernelSize)    
    borderSize = int((kernelSize-1)/2)
    matrix = getBaseMatrix(borderType, borderSize)
    
    applyConvolution(matrix, kernel/1000000, borderSize, borderType)

def rayleigh(borderType):
    kernel = (ray13/100000000) * ray13Factor
    matrix = getBaseMatrix(borderType, 1)

    applyConvolution(matrix, kernel, 1, borderType)

def getSobelGradient(gradientX, gradientY):
    shape = np.shape(gradientX)
    
    gradient = np.empty(shape[0])
    
    for i in range(0, shape[0]):
        for j in range(0, shape[1]):
            gradient[i, j] = abs(gradientX[i, j]) + abs(gradientY[i, j])
    return gradient

def sobel():
    #FIXME: dont process the original image, process the one who is filtered
    filteredImage = dicom.read_file(lstFilesDCM[file_cb.current()]).pixel_array
    
    gradientX = applyConvolution(filteredImage, sobelx, 1, 3)
    gradientY = applyConvolution(filteredImage, sobely, 1, 3)
    
    gradient = getSobelGradient(gradientX, gradientY)
    plotImages(gradient)

# Menu options
def applyFunction():
    function = functions_cb.get()    
    if function == 'Histogram':
        hist()
    elif function == 'Average filter':        
        kernelSize = simpledialog.askinteger("Kernel Size", "Digit the kernel size\n", parent=root, minvalue=3, maxvalue=999)       
        if kernelSize % 2 == 0:
            messagebox.showinfo("Warning","The kernel size must be an odd number")
            return
        
        borderType = simpledialog.askinteger("Border Type", "Digit the border type\n\n1. Mirror\n\n2. Replicate\n\n3. Ignore\n", parent=root, minvalue=1, maxvalue=3)
        averageFilter(kernelSize, borderType)
    elif function == 'Gaussian filter':
        sigma = simpledialog.askfloat("Sigma", "Choose the desired standard deviation\n\n(0.5, 1.0, 1.5)\n", parent=root)       
        if sigma != 0.5 and sigma != 1.0 and sigma != 1.5:
            messagebox.showinfo("Warning","Choose one of the options")
            return

        kernelSize = simpledialog.askinteger("Kernel Size", "Choose a kernel size\n\n(3, 5, 7, 11)\n", parent=root)       
        if kernelSize % 2 == 0 or kernelSize < 3 or kernelSize > 11:
            messagebox.showinfo("Warning","Choose one of the options")
            return

        borderType = simpledialog.askinteger("Border Type", "Digit the border type\n\n1) Mirror\n\n2) Replicate\n\n3) Ignore\n", parent=root, minvalue=1, maxvalue=3)
        gaussianFilter(sigma, kernelSize, borderType)
    elif function == 'Rayleigh filter':
        borderType = simpledialog.askinteger("Border Type", "Digit the border type\n\n1) Mirror\n\n2) Replicate\n\n3) Ignore\n", parent=root, minvalue=1, maxvalue=3)
        rayleigh(borderType)   
    elif function == 'Sobel':
        sobel()
    else:
        messagebox.showinfo("Error", "Function not found")    

#TK components
root = tk.Tk()
root.title("Medical Imaging")
root.configure(background='pink')
root.geometry('%dx%d+%d+%d' % (root.winfo_screenwidth(), root.winfo_screenheight(), 0, 0))

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
functions_cb["values"] = ['Histogram', 'Average filter', 'Gaussian filter', 'Rayleigh filter', 'Sobel']

apply_bt = tk.Button(files_fr, text="Apply", command=applyFunction, bg='white', state=DISABLED)
apply_bt.grid(row=1, column=1, pady=10, padx=5)

text_fr = Frame(root)
text_fr.configure(background='pink')

info_text = tk.Text(text_fr, width = 90, height = 11)
info_text.pack(pady=20)

images_fr = Frame(root)
images_fr.configure(background='pink')

left_image_fr = Frame(images_fr)
right_image_fr = Frame(images_fr)

f = Figure()

canvas = FigureCanvasTkAgg(f, master=left_image_fr)
canvas2 = FigureCanvasTkAgg(f, master=right_image_fr)

root.mainloop()