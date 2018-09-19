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


lstFilesDCM = []      
def headerInfo():   
    info_text.delete('1.0', END)        
    processImage()

def selectFolder():    
    folder =  filedialog.askdirectory()
    loadFiles(folder)

def loadFiles(folder):
    resetSelector()
    lstFilenames = []
    for dirName, subdirList, fileList in os.walk(folder):
        for filename in fileList:
            if "" in filename.lower():
                lstFilesDCM.append(os.path.join(dirName,filename))
                lstFilenames.append(filename)
    
    cb["values"] = lstFilenames    
    files_fr.pack() 

def resetSelector():
    global lstFilesDCM
    lstFilesDCM.clear()
    info_text.delete('1.0', END)
    cb.set("Select a DICOM file")    
    
def showHeaderInfo(header):
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
    RefDs = dicom.read_file(lstFilesDCM[cb.current()])
    rows = int(RefDs.Rows)
    columns = int(RefDs.Columns)
    
    intensity = [0]*65536
    
    for i in range(rows):
        for j in range(columns):
            intensity[RefDs.pixel_array[i,j]]=intensity[RefDs.pixel_array[i,j]]+1
            
    intensity = np.asarray(intensity)
    plt.plot(intensity)
    plt.show()
    
            
def processImage():
    global canvas
    RefDs = dicom.read_file(lstFilesDCM[cb.current()])
    showHeaderInfo(RefDs)
    
    plt.set_cmap(plt.gray())
    f = Figure()
    a = f.add_subplot(111)    
    a.imshow(np.flipud(RefDs.pixel_array))     
    canvas.get_tk_widget().destroy()
    canvas = FigureCanvasTkAgg(f, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack()    
    

#TK components
root = tk.Tk()
root.title("Medical Imaging")
root.configure(background='black')
root.geometry("{0}x{1}+0+0".format(root.winfo_screenwidth(), root.winfo_screenheight()))

selectFolder_fr = Frame(root)
selectFolder_fr.configure(background='black')
selectFolder_fr.pack()

selectFolder_bt = tk.Button(selectFolder_fr, text="Select folder", command=selectFolder, bg='white')
selectFolder_bt.pack(pady=5)

files_fr =Frame(root)
files_fr.configure(background='black')

cb = ttk.Combobox(files_fr, state='readonly')
cb.set("Select a DICOM file")
cb.pack(padx=20, pady=5)

process_bt = tk.Button(files_fr, text="Process file", command=headerInfo, bg='white')
process_bt.pack(pady=5)

his_bt = tk.Button(files_fr, text="Show histogram", command=hist, bg='white')
his_bt.pack(pady=5)
   
info_text = tk.Text(files_fr, width = 90, height = 11)
info_text.pack(pady=20)

f = Figure()
canvas = FigureCanvasTkAgg(f, master=root)

root.mainloop()