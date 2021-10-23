# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 15:06:02 2021

@author: singh
"""
import os
import librosa
import keras
import numpy as np
import IPython.display as ipd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.utils import to_categorical
from keras.layers import Input, Flatten, Dropout, Activation
from keras.layers import Conv1D, MaxPooling1D
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from tensorflow.keras import optimizers


model=keras.models.load_model('testing12_model.h5')
model.summary()

def get_label(predictions):
  # 0 = neutral, 1 = calm, 2 = happy, 3 = sad, 4 = angry, 5 = fearful, 6 = disgust, 7 = surprised
  label =[]
  for prediction in predictions:
    if prediction == "0":
        label.append("Neutral")
    elif prediction == 1:
        label.append("Calm")
    elif prediction == 2:
        label.append("Happy")
    elif prediction == 3:
        label.append("Sad")
    elif prediction == 4:
        label.append("Angry")
    elif prediction == 5:
        label.append("Fearful")
    elif prediction == 6:
        label.append("Disgust")
    elif prediction == 7:
        label.append("Suprsied")
    else:
        label.append("None")
  return label


def ser(filepath):
  #filepath=np.asarray(filepath)
  #rootname = os.path.dirname(os.path.realpath(__file__))
  arr =[]
  for f ,acc,name in filepath:
    #print(f+" --- "+str(acc))
    lst = []
    filename= f
    X, sample_rate = librosa.load(filename, res_type='kaiser_fast',duration=4,sr=22050*2,offset=0.5)
    #plot_time_series(X,'Raw') 
    #ipd.Audio(X, rate=sample_rate)
    # Trim the silence voice
    aa , bb = librosa.effects.trim(X, top_db=30)
    # Silence trimmed Sound by librosa.effects.trim()
    #plot_time_series(aa,'Trim')
    #ipd.Audio(aa, rate=sample_rate)
    mfccs = np.mean(librosa.feature.mfcc(y=aa, sr=sample_rate,n_mfcc=40).T, axis=0)
    arr = mfccs, acc
    lst.append(arr)
    x, y = zip(*lst)
    x =np.asarray(x)
    x=np.expand_dims(x, axis=2)
    prediction = model.predict_classes(x)
    prediction=prediction.astype(int)
    if(acc != prediction[0]):
        print(f)


def toemotion(pred):
        """
        Method to convert the predictions (int) into human readable strings.
        """
        
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
  lst = []
  for f in sorted(os.listdir(dirpath)):
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
    if(prediction[0]==acc):
        actualemotion = toemotion(acc)
        predictedemotion = toemotion(prediction[0])
        print(filename+" ---- " + actualemotion + " ---- " + predictedemotion)

EXAMPLE_DIR= os.path.join(os.path.dirname(os.path.realpath(__file__)),'examples')
#print(EXAMPLE_DIR)
#serDEMO(EXAMPLE_DIR)

#filename = '/content/drive/MyDrive/dataset/Actor_02/03-02-06-02-01-02-02.wav'
#X, sample_rate = librosa.load(filename, res_type='kaiser_fast',duration=4,sr=22050*2,offset=0.5)
#ipd.Audio(X, rate=sample_rate)

selfile = []
def serfol(path):
  selfile.clear()
  for d in sorted(os.listdir(path)):
    if d == 'Actor_26': continue
    elif d == 'Actor_25': continue
    else:
      #filepath=np.asarray(filepath)
      dir = os.path.join(path,d)
      #print(dir)
      
      for f in os.listdir(dir):
        lst = []
        filename = os.path.join(dir,f)
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
        if(prediction[0] == acc):
          pair = filename , acc, f 
          selfile.append(pair)
          print(filename+ " --------- " +toemotion(prediction))
          
#TEST_DIR= os.path.join(os.path.dirname(os.path.realpath(__file__)),'dataset1')
#print(TEST_DIR)
#serfol(TEST_DIR)

#print(len(selfile))
#print(selfile)


import pickle
#with open("test.txt", "wb") as fp:   #Pickling
    #pickle.dump(selfile, fp)

with open("test.txt", "rb") as fp:   # Unpickling
    b = pickle.load(fp)
  
ser(b)
print(len(b))  


import shutil
TESTING_DIR= os.path.join(os.path.dirname(os.path.realpath(__file__)),'Testing')
for f,acc ,a in b:
    name = a
    shutil.copy(f, os.path.join(TESTING_DIR,name))
    
def serDEMO(dirpath):
  for f in os.listdir(dirpath):
    # print(f)
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
    
    
serDEMO(TESTING_DIR)