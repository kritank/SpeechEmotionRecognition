# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 18:30:54 2021

@author: singh
"""

import os
import librosa
import keras
import numpy as np
import warnings
from playsound import playsound
warnings.filterwarnings("ignore")
DEMO_DIR= os.path.join(os.path.dirname(os.path.realpath(__file__)),'Testing')


def toemotion(pred):
        label_conversion = {'0': 'neutral',
                            '1': 'calm',
                            '2': 'happy',
                            '3': 'sad',
                            '4': 'angry',
                            '5': 'fearful',
                            '6': 'disgust',
                            '7': 'surprised'}

        for key, value in label_conversion.items():
            if int(key) == pred:
                label = value
        return label



def serDEMO(dirpath):
  for f in os.listdir(dirpath):
    lst = []
    filename= os.path.join(dirpath,f)
    X, sample_rate = librosa.load(filename, res_type='kaiser_fast',duration=4,sr=22050*2,offset=0.5)
    aa , bb = librosa.effects.trim(X, top_db=30)
    mfccs = np.mean(librosa.feature.mfcc(y=aa, sr=sample_rate,n_mfcc=40).T, axis=0)
    acc = int(f[7:8]) - 1
    arr = mfccs, acc
    lst.append(arr)
    x, y = zip(*lst)
    x =np.asarray(x)
    x=np.expand_dims(x, axis=2)
    prediction = model.predict_classes(x)
    prediction=prediction.astype(int)
    actualemotion = toemotion(acc)
    predictedemotion = toemotion(prediction[0])
    print(filename+" ---- " + actualemotion + " ---- " + predictedemotion)
    
    

def demo():
    for f in os.listdir(DEMO_DIR):
        lst=[]    
        filename = os.path.join(DEMO_DIR,f)
        X, sample_rate = librosa.load(filename, res_type='kaiser_fast',duration=4,sr=22050*2,offset=0.5)
        aa , bb = librosa.effects.trim(X, top_db=30)
        mfccs = np.mean(librosa.feature.mfcc(y=aa, sr=sample_rate,n_mfcc=40).T, axis=0)
        acc = int(f[7:8]) - 1
        arr = mfccs, acc
        lst.append(arr)
        x, y = zip(*lst)
        x =np.asarray(x)
        x=np.expand_dims(x, axis=2)
        prediction = model.predict_classes(x)
        prediction=prediction.astype(int)
        #actualemotion = toemotion(acc)
        predictedemotion = toemotion(prediction[0])
        playsound(filename) 
        print('[INFO] Played: '+ f + " ----> "+predictedemotion) 






# Loading the model
print("[INFO] Loading the model")
model=keras.models.load_model('testing12_model.h5')
print("[INFO] Model loading")
print("[INFO] Model Summary")
model.summary()
print("[INFO] INTIATING DEMO")
demo()



