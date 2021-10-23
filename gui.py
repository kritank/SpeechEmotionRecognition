# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 21:01:33 2021

@author: singh
"""

# Python program to create 
# a file explorer in Tkinter

# import all components
# from the tkinter library
from tkinter import *

# import filedialog module
from tkinter import filedialog

# Function for opening the 
# file explorer window
from playsound import playsound


import os
import librosa
import keras
import numpy as np
import warnings
from playsound import playsound
warnings.filterwarnings("ignore")

model=keras.models.load_model('testing12_model.h5')

filename = []

def browseFiles():
    filename.clear()
    f = filedialog.askopenfilename(initialdir = "/",
                                          title = "Select a File",
                                          filetypes = (
                                              ("all files",
                                                        "*.*"),
                                              ("Text files",
                                                        "*.txt*")
                                                       ))
    print(f)
    if len(f) == 0:
        return
    else:
        filename.append(f)
        # Change label contents
        label_file_explorer.configure(text="File Opened: "+f)
        label_file_emotion.configure(text="First select a file")
	
def exit():
    print("dndj")
    
def play():
    #print(filename)
    if len(filename) !=0:
        playsound(filename[0])  
    


def toemotion(pred):
        label_conversion = {'0': 'Neutral',
                            '1': 'Calm',
                            '2': 'Happy',
                            '3': 'Sad',
                            '4': 'Angry',
                            '5': 'Fearful',
                            '6': 'Disgust',
                            '7': 'Surprised'}

        for key, value in label_conversion.items():
            if int(key) == pred:
                label = value
        return label



def predict():
    if len(filename) !=0:
        lst=[]    
        file = filename[0]
        X, sample_rate = librosa.load(file, res_type='kaiser_fast',duration=4,sr=22050*2,offset=0.5)
        aa , bb = librosa.effects.trim(X, top_db=30)
        mfccs = np.mean(librosa.feature.mfcc(y=aa, sr=sample_rate,n_mfcc=40).T, axis=0)
        acc = 1
        arr = mfccs, acc
        lst.append(arr)
        x, y = zip(*lst)
        x =np.asarray(x)
        x=np.expand_dims(x, axis=2)
        prediction = model.predict_classes(x)
        prediction=prediction.astype(int)
        #actualemotion = toemotion(acc)
        predictedemotion = toemotion(prediction[0])
        print('[INFO] Played: '+ file + " ----> "+predictedemotion)
        # Change label contents
        label_file_emotion.configure(text="Observed Emotion : "+predictedemotion)
																								
# Create the root window
window = Tk()

# Set window title
window.title('SPEECH EMOTION RECOGNITION')

# Set window size
window.geometry("700x334")

#Set window background color
window.config(background = "gray25")


import tkinter.font as font
buttonFont = font.Font(family='Helvetica', size=16, weight='bold')	

# Create a File Explorer label
label_file_explorer = Label(window,
							text = "No file has been selected",
							width = 100, height = 4, 
							fg = "blue", bg='lavender') 
label_file_explorer.config(anchor=CENTER)



label_file_emotion = Label(window, 
							text = "First select a file",
							width = 100, height = 4, 
							fg = "red", bg='peach puff')
label_file_emotion.config(anchor=CENTER)

button_explore = Button(window, 
						text = "Select a sample",
						command = browseFiles,width=100,height=4,bd=0 , bg="cyan", activebackground='gray43' , fg ='blue') 

button_exit = Button(window, 
					text = "Exit",
					command = exit) 
button_play = Button(window, 
					text = "Play",
					command = play,width=100,height=4 , bd=0,bg="spring green", activebackground='gray43' , fg ='green') 

button_predict = Button(window, 
					text = "Predict",
					command = predict,width=100,height=4,bd=0 , bg="tan1", activebackground='gray43' , fg ='OrangeRed2') 
# Grid method is chosen for placing
# the widgets at respective positions 
# in a table like structure by
# specifying rows and columns
label_file_explorer.grid(column = 1, row = 1)
label_file_emotion.grid(column = 1, row = 2)

button_explore.grid(column = 1, row = 3)

#button_exit.grid(column = 1,row = 4)
button_play.grid(column = 1,row = 4)
button_predict.grid(column = 1,row = 5)

# Let the window wait for any events
window.mainloop()
