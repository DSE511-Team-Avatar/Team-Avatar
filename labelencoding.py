##### Data cleaning by Pragya####
#importing the packages
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

#loading and displaying the data
housing = pd.read_csv('housing.csv')
housing.head()

#looking at the type of data
housing.info()

#creating instance for label encoder
le = LabelEncoder()
#Asssigning neumerical values to ocean proximity
housing["ocean_proximity"]= le.fit_transform(housing["ocean_proximity"])

#printing data
housing
