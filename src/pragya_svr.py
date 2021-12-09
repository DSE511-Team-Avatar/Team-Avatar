#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 09:48:37 2021

@author: pragyakandel
"""
#Importing the base libraries

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.style.use(['ggplot']) #use ggplot style

#Importing the sklearn packages

from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import mean_absolute_error,mean_squared_error

# Defining the function for Support Vector Regression

def pragya_svr(housing, housing_labels, housing_t, housing_labels_t):
    # Create SVR model 
    clf  = svm.SVR()
    #Calculating the wall time for the training set
   
    #%%time
    clf.fit(housing, housing_labels)   # Fit to the Training set
    
    #Printing the model scores for trainning and testing set
    print("SVR model score Training Set:", clf.score(housing, housing_labels))
    print("SVR model score Testing Set:", clf.score(housing_t, housing_labels_t))
    
    #Calculating the wall time for testing
    #%%time
    prediction = clf.predict(housing_t) ## SVR Model Evaluation
    
    #Printing the errors on testing set
    print('MAE:', metrics.mean_absolute_error(housing_labels_t,prediction))
    print('MSE:', metrics.mean_squared_error(housing_labels_t, prediction))
    print('RMSE:', np.sqrt(metrics.mean_squared_error(housing_labels_t, prediction)))
    ###############################################################################################
    print(" Hyperparameter tuning")
    #Model for hyperparameter tuning.
    parameters = {'kernel': ('linear', 'rbf','poly'), 'C':[1.5, 10],'gamma': [1e-7, 1e-4],'epsilon':[0.1,0.2,0.5,0.3]}
    clf_tuned = GridSearchCV(clf, parameters)
    
    #Wall time for hyperparamter tuning for training set
    #%%time
    clf_tuned.fit(housing,housing_labels)  # fit on the model (tuned)

    #Checking for best parameters
    print(clf_tuned.best_params_)

    # Printing the for best score
    print(clf_tuned.best_score_)
    
    # Wall time for testing set
    #%%time
    tuned_pred = clf_tuned.best_estimator_.predict(housing_t)  # put the tuned model on testing set
    
    #Printing errors after hyperparameter tuning
  
    print('MAE:', metrics.mean_absolute_error(housing_labels_t,tuned_pred))
    print('MSE:', metrics.mean_squared_error(housing_labels_t, tuned_pred))
    print('RMSE:', np.sqrt(metrics.mean_squared_error(housing_labels_t, tuned_pred)))
    
    ######################################################################
    print(" Hyperparameter tuning reduced MAE, MSE and RMSE in this dataset.")
    
    #Calculating the cross validation score
    cvs = cross_val_score(clf_tuned ,housing, housing_labels, scoring = "neg_mean_squared_error", cv=5)
    cvs_sqrt = np.sqrt(-cvs)
    cvs_tuned_mean = np.mean(cvs_sqrt) 
    print("Cross validation score after tuning : ", cvs_tuned_mean)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
