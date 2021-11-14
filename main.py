# DSE511: Introduction to Data Science and Computing

#Import the basic important libraries
import numpy as np
import pandas as pd


#Extract data from the file:
data = pd.read_csv('Data/housing.csv')
data.head()


print('There are {} rows and {} columns in train'.format(data.shape[0],data.shape[1]))



#Split the data first and do all feature transformations after the test_train splitting on the train set only to avoid data leakage

from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(data, test_size=0.2, random_state=1)

print("Training Data", len(train_set))
print("Testing Data", len(test_set))
