# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 21:01:59 2021

@author: singh

This files creates the X and y features in joblib to be used by the predictive models.
"""

import os
import time
import joblib
import librosa
import numpy as np


TRAINING_FILES_PATH = "C:\\Users\\singh\\OneDrive\\Desktop\\Speech Emotion\\dataset"
#"C:\\Users\\singh\\Downloads\\emotion_recognition_cnn-master\\dataset\\"
SAVE_DIR_PATH = "C:\\Users\\singh\\OneDrive\\Desktop\\Speech Emotion"
#"C:\\Users\\singh\\Downloads\\emotion_recognition_cnn-master\\"



class CreateFeatures:

    @staticmethod
    def features_creator(path, save_dir) -> str:
        """
        This function creates the dataset and saves both data and labels in
        two files, X.joblib and y.joblib in the joblib_features folder.
        With this method, you can persist your features and train quickly
        new machine learning models instead of reloading the features
        every time with this pipeline.
        """

        lst = []

        start_time = time.time()

        for subdir, dirs, files in os.walk(path):
            for file in files:
                try:
                    # Load librosa array, obtain mfcss, store the file and the mcss information in a new array
                    X, sample_rate = librosa.load(os.path.join(subdir, file),
                                                  res_type='kaiser_fast')
                    aa , bb = librosa.effects.trim(X, top_db=30)
                    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate,
                                                         n_mfcc=40).T, axis=0)
                    # The instruction below converts the labels (from 1 to 8) to a series from 0 to 7
                    # This is because our predictor needs to start from 0 otherwise it will try to predict also 0.
                    file = int(file[7:8]) - 1
                    arr = mfccs, file
                    lst.append(arr)
                # If the file is not valid, skip it
                except ValueError as err:
                    print(err)
                    continue

        print("--- Data loaded. Loading time: %s seconds ---" % (time.time() - start_time))

        # Creating X and y: zip makes a list of all the first elements, and a list of all the second elements.
        X, y = zip(*lst)

        # Array conversion
        X, y = np.asarray(X), np.asarray(y)

        # Array shape check
        print(X.shape, y.shape)

        # Preparing features dump
        X_name, y_name = 'x6.joblib', 'y6.joblib'

        joblib.dump(X, os.path.join(save_dir, X_name))
        joblib.dump(y, os.path.join(save_dir, y_name))

        return "Completed"


if __name__ == '__main__':
    print('Routine started')
    FEATURES = CreateFeatures.features_creator(path=TRAINING_FILES_PATH, save_dir=SAVE_DIR_PATH)
    print('Routine completed.')
