#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 16:35:10 2021

@author: pragyakandel
"""

#### importing the libraries####
from sklearn.svm import SVR
import matplotlib as mpl
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.metrics import classification_report
mpl.style.use(['ggplot']) #use ggplot style
from sklearn.metrics import mean_absolute_error,mean_squared_error

# First, we will split between features and target value (labels) for train set

housing = train.drop("median_house_value", axis=1) # drop labels for training set
housing_labels = train["median_house_value"].copy()

housing
