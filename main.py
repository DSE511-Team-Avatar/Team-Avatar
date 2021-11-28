# DSE511: Introduction to Data Science and Computing - Fall 2021
# California Housing Prices (Final Project)
# Team-Avatar

#*************************************************************************************************
#....> Importing modules
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
#....................................................................................................
print('Modules are imported')  #....> If prints, this means that the modules are imported correctly
#****************************************************************************************************

### Read in all our data

data = pd.read_csv('Data/housing.csv')  #....> Importing dataframe
rows,columns = df.shape
print("Dataframe number of rows: ", rows)
print("Dataframe number of columns: ", columns)



#print('There are {} rows and {} columns in train'.format(data.shape[0],data.shape[1]))



#Split the data first and do all feature transformations after the test_train splitting on the train set only to avoid data leakage
train_set, test_set = train_test_split(data, test_size=0.2, random_state=1)

print("Training Data", len(train_set))
print("Testing Data", len(test_set))
