# DSE511: Introduction to Data Science and Computing - Fall 2021
# California Housing Prices (Final Project)
# Team-Avatar

# Importing libraries
import numpy as np
import pandas as pd
import time
import seaborn as sns
import matplotlib.pyplot as plt

# Extract raw data from Data folder:
data = pd.read_csv('Data/housing.csv')



print('There are {} rows and {} columns in train'.format(data.shape[0],data.shape[1]))


#Split the data first and do all feature transformations after the test_train splitting on the train set only to avoid data leakage

from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(data, test_size=0.2, random_state=1)

print("Training Data", len(train_set))
print("Testing Data", len(test_set))
