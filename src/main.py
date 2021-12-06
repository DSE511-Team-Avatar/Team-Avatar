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
import matplotlib as mpl

#....> Preventing warnings
import warnings      #....> Prevent from printing the warning of plotting
warnings.simplefilter(action="ignore", category=FutureWarning)

print('Modules in main.py are imported\n') #....> If prints, this means that the modules in main.py are imported correctly


#****************************************************************************************************
#                                 Insert dataframe
#****************************************************************************************************
df = pd.read_csv('../Data/housing.csv')  #....> Importing dataframe

#****** Notice:
# ...> Use the following dataframe for Capped
df_Capped = df
# ...> Use the following dataframe for No Capped
df_NoCapped = df[df['median_house_value'] < 490000]


#****************************************************************************************************
#                                Data pre-processing 
#****************************************************************************************************
#....> Import pre-processing module
from preprocessing import *

housing, housing_labels, housing_t, housing_labels_t = preprocessing(df_Capped)

    
#****************************************************************************************************
#                             Try our Functions (models)
#****************************************************************************************************

