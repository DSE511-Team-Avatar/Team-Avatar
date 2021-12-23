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
# ...> For Capped
df_Capped = df
# ...> For No Capped
df_NoCapped = df[df['median_house_value'] < 490000]


#****************************************************************************************************
#                                Data pre-processing 
#****************************************************************************************************
#....> Import pre-processing module
from preprocessing import *

#****** Notice:
# ...> For Capped use:      df_Capped
# ...> For No Capped use:   df_NoCapped
housing, housing_labels, housing_t, housing_labels_t = preprocessing(df_Capped)

    
#****************************************************************************************************
#                             Try our Functions (models)
#****************************************************************************************************

#....> To call Linear Regression uncomment these two lines
#from only_function_linreg_Albina import *
#linreg_Albina(housing, housing_labels, housing_t, housing_labels_t)


#....> To call Lasso uncomment these two lines:
#from only_function_lasso_Albina import *
#lasso_Albina(housing, housing_labels, housing_t, housing_labels_t)


#....> To call Ridge uncomment these two lines:
#from only_function_ridge_Albina import *
#ridge_Albina(housing, housing_labels, housing_t, housing_labels_t)


#....> To call ElasticNet uncomment these two lines:
#from only_function_elasticnet_Albina import *
#elasticnet_Albina(housing, housing_labels, housing_t, housing_labels_t)


#....> To call Random Forest uncomment these two lines:
#from isi_rf_function import *
#isi_random_forest(housing, housing_labels, housing_t, housing_labels_t)


#....> To call Decision Tree Regressor uncomment these two lines:
#from ehsan_dtr import *
#ehsan_dtr(housing, housing_labels, housing_t, housing_labels_t)



#....> To call Support Vector Regression uncomment these two lines: 
#from pragya_svr import *
#pragya_svr(housing, housing_labels, housing_t, housing_labels_t)

