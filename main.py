# DSE511: Introduction to Data Science and Computing - Fall 2021
# California Housing Prices (Final Project)
# Team-Avatar

#*************************************************************************************************
#*************************************************************************************************
#                       ....> Importing modules <....
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time


#....> AI/ML modules
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error 
from sklearn.model_selection import GridSearchCV
from sklearn import metrics


#....> Preventing warnings
import warnings      #....> Prevent from printing the warning of plotting
warnings.simplefilter(action="ignore", category=FutureWarning)
#....................................................................................................
print('Modules are imported')  #....> If prints, this means that the modules are imported correctly

#****************************************************************************************************
#****************************************************************************************************
#                         ....> Read dataset <....

df = pd.read_csv('Data/housing.csv')  #....> Importing dataframe
rows, columns = df.shape
print("Dataframe number of rows: ", rows)
print("Dataframe number of columns: ", columns)

#****************************************************************************************************
#****************************************************************************************************
#                         ....> Data pre-processing <....
from preprocessing import *

#....> Show the structure
show_structure(df)

#....> Add some more features
add_more_features(df)
show_structure(df)

#...> Handling categorical feature(ocean_proximity)
handling_categorical(df)
show_structure(df)

splitting_dataset(df)

