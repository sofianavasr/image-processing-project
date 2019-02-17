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
baseImage = np.zeros(512, float)
processedImage = np.zeros(512, float)
tones = 65536

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
    global lstFilesDCM, baseImage
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
        
def hist(image):    
    size = np.shape(image)
    rows = int(size[0])   
    columns = int(size[1])
    intensity = np.zeros(tones)
    
    for i in range(rows):
        for j in range(columns):
            intensity[image[i,j]]=intensity[image[i,j]]+1
            
    plt.plot(intensity)
   
    fig = plt.gcf()
    fig.canvas.set_window_title('Histogram')       
    
    plt.show()   


def plotImages(newImage):
    global canvas, canvas2
    plt.set_cmap(plt.gray())
    f = Figure(figsize=(80,80))
    a = f.add_subplot(121)  
    a2 = f.add_subplot(122)

    a.imshow(baseImage)
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
    global baseImage, processImage
    text_fr.pack()   
    images_fr.pack(fill='both', expand=True)
    left_image_fr.pack(fill='both', expand=True)
    right_image_fr.pack(fill='both', expand=True)
  
    apply_bt.configure(state=NORMAL)

    imageInfo = dicom.read_file(lstFilesDCM[file_cb.current()])
    showHeaderInfo(imageInfo)
    baseImage = np.copy(imageInfo.pixel_array)
    processImageImage = np.copy(imageInfo.pixel_array)

    plotImages(baseImage)
        
def getBaseMatrix(image, borderType, borderSize):    
    if borderType == 1:
        return np.pad(image, borderSize, mode='symmetric')
    elif borderType == 2:
        return np.pad(image, borderSize, mode='edge')
    elif borderType == 3:
        return image
    
def applyConvolution(matrix, kernel, borderSize, borderType):
    shape = np.shape(matrix)
    rowsLimit = shape[0] - borderSize
    columnsLimit = shape[1] - borderSize
    convMatrix = np.copy(matrix)
    
    for i in range(borderSize, rowsLimit):
        for j in range(borderSize, columnsLimit):
            submatrix = matrix[i-borderSize:i+borderSize+1:1,j-borderSize:j+borderSize+1:1]        
            convMatrix[i,j] = np.sum(np.multiply(submatrix, kernel))
    
    if borderType == 3:
        finalMatrix = convMatrix
    else:
        finalMatrix = convMatrix[borderSize:rowsLimit+1:1, borderSize:columnsLimit+1:1]   
    
    """ plt.imshow(convMatrix)
    plt.gcf().canvas.set_window_title('Image Filtering')     
    plt.show() """
    
    return finalMatrix
   
def averageFilter(image, kernelSize, borderType):        
    global processImage

    kernel = np.ones((kernelSize, kernelSize))
    kernelFactor = 1/np.sum(kernel)    
    borderSize = int((kernelSize-1)/2)
    matrix = getBaseMatrix(image, borderType, borderSize)

    finalMatrix = applyConvolution(matrix, kernel*kernelFactor, borderSize, borderType)
    
    plotImages(finalMatrix)
    processImage = finalMatrix

#def mediana(image, kernelSize, borderType):




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

def gaussianFilter(image, sigma, kernelSize, borderType):
    global processImage

    kernel = getGaussianKernel(sigma, kernelSize)    
    borderSize = int((kernelSize-1)/2)
    matrix = getBaseMatrix(image, borderType, borderSize)
    
    finalMatrix = applyConvolution(matrix, kernel/1000000, borderSize, borderType)
    
    plotImages(finalMatrix)
    processImage = finalMatrix

def rayleigh(image, borderType):
    global processImage
    kernel = (ray13/100000000) * ray13Factor
    matrix = getBaseMatrix(image, borderType, 1)

    finalMatrix = applyConvolution(matrix, kernel, 1, borderType)

    plotImages(finalMatrix)
    processImage = finalMatrix

def sobel(image):           
    global processImage

    shape = np.shape(image)    
    gradient = np.copy(image)

    for i in range(1, shape[0] - 1):
        for j in range(1, shape[1] - 1):
            submatrix = image[i-1:i+1+1:1,j-1:j+1+1:1]        
            gradient[i,j] = np.absolute(np.sum(np.multiply(submatrix, sobelx))) + np.absolute(np.sum(np.multiply(submatrix, sobely)))
        
    plotImages(gradient)
    processImage = gradient

# Menu options
def applyFunction():
    currentImage = np.copy(processImage)
    if image_cb.get() == 'Original':
        currentImage = np.copy(baseImage)
    
    function = functions_cb.get()    
    if function == 'Histogram':
        hist(currentImage)
    elif function == 'Average filter':        
        kernelSize = simpledialog.askinteger("Kernel Size", "Digit the kernel size\n", parent=root, minvalue=3, maxvalue=999)       
        if kernelSize % 2 == 0:
            messagebox.showinfo("Warning","The kernel size must be an odd number")
            return
        
        borderType = simpledialog.askinteger("Border Type", "Digit the border type\n\n1. Mirror\n\n2. Replicate\n\n3. Ignore\n", parent=root, minvalue=1, maxvalue=3)
        averageFilter(currentImage, kernelSize, borderType)
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
        gaussianFilter(currentImage, sigma, kernelSize, borderType)
    elif function == 'Rayleigh filter':
        borderType = simpledialog.askinteger("Border Type", "Digit the border type\n\n1) Mirror\n\n2) Replicate\n\n3) Ignore\n", parent=root, minvalue=1, maxvalue=3)
        rayleigh(currentImage, borderType)   
    elif function == 'Sobel':
        sobel(currentImage)
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
file_cb.grid(row=0, column=1, padx=5)

process_bt = tk.Button(files_fr, text="Process", command=processImage, bg='white')
process_bt.grid(row=0, column=2, padx=5)

image_cb = ttk.Combobox(files_fr, state='readonly')
image_cb.set("Choose an image")
image_cb["values"] = ['Original', 'Processed']
image_cb.grid(row=1, column=0, pady=10, padx=5)

functions_cb = ttk.Combobox(files_fr, state='readonly')
functions_cb.set("Select function")
functions_cb.grid(row=1, column=1, pady=10)
functions_cb["values"] = ['Histogram', 'Average filter', 'Gaussian filter', 'Rayleigh filter', 'Sobel']

apply_bt = tk.Button(files_fr, text="Apply", command=applyFunction, bg='white', state=DISABLED)
apply_bt.grid(row=1, column=2, pady=10, padx=5)

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