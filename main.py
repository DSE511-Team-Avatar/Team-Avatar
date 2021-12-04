# DSE511: Introduction to Data Science and Computing - Fall 2021
# California Housing Prices (Final Project)
# Team-Avatar

#*************************************************************************************************
#                                 Importing basic modules
# ************************************************************************************************                      
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#....> Preventing warnings
import warnings      #....> Prevent from printing the warning of plotting
warnings.simplefilter(action="ignore", category=FutureWarning)

print('Modules in main.py are imported\n') #....> If prints, this means that the modules in main.py are imported correctly





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

#....> Import pre-processing module
from preprocessing import *


#....> Show the structure
show_structure(df)

#....> Add some more features
df = adding_additional_features(df)
show_structure(df)

#...> Handling categorical feature(ocean_proximity)
df = handling_categorical_attributes(df)
show_structure(df)

#....> Data Splitting
train_set, test_set = data_splitting(df)

# Check for NaN
#....> Check the train_set
check_null(train_set)
#....> Check the test_set
check_null(test_set)


# Check the Percentage of miising value
#....> Check the train_set
missing_percentage(train_set)
#....> Check the test_set
missing_percentage(test_set)


# Data cleaning 
train_set, test_set = data_cleaning(df,train_set, test_set)

# show_structure(df)
show_structure(train_set)
show_structure(test_set)




