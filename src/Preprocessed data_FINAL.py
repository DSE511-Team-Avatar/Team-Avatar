#!/usr/bin/env python
# coding: utf-8

# In[23]:


# Updated by: Albina Jetybayeva
# This jupyter notebook contains the preprocessing step from Isi, Albina, Pragya and Ehsan together.

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split


# In[24]:


# Getting the raw data
df = pd.read_csv('housing.csv') # Notice: Raw data is in the Data folder
df


# In[25]:


print('There are {} rows and {} columns in train'.format(df.shape[0],df.shape[1]))


# In[26]:


#Include additional features that might be important
df["rooms_per_household"] = df["total_rooms"]/df["households"]
df["bedrooms_per_room"] = df["total_bedrooms"]/df["total_rooms"]
df["population_per_household"]=df["population"]/df["households"]
df


# ## Handling categorical attributes

# In[27]:


# Creating instance for label encoder.
le = LabelEncoder()

# Asssigning numerical values to ocean proximity.
df["ocean_proximity"]= le.fit_transform(df["ocean_proximity"])
print(df["ocean_proximity"])
df["ocean_proximity"].value_counts()


# ## Split the data

# In[28]:


# Splitting the data into training and testing sets.
train_set1, test_set1 = train_test_split(df, test_size=0.2, random_state=1)
print("Training Data", len(train_set1))
print("Testing Data", len(test_set1))


# ## Data cleaning on train and test

# In[29]:


# Observing missing values
missing_values_count = train_set1.isnull().sum()
missing_values_count[:]

total_cells   = np.product(train_set1.shape)
total_missing = missing_values_count.sum()
percent_missing = (total_missing/total_cells)*100
print('Percent of data that is missing:', percent_missing)

imputer = SimpleImputer(strategy = "median")
housing_numerical_attributes = train_set1.drop("ocean_proximity", axis = 1)
imputer.fit(housing_numerical_attributes)  
X = imputer.transform(housing_numerical_attributes)


# In[30]:


# Observing missing values
missing_values_count = test_set1.isnull().sum()
missing_values_count[:]

total_cells   = np.product(test_set1.shape)
total_missing = missing_values_count.sum()
percent_missing = (total_missing/total_cells)*100
print('Percent of data that is missing:', percent_missing)

imputer = SimpleImputer(strategy = "median")
housing_numerical_attributes1 = test_set1.drop("ocean_proximity", axis = 1)
imputer.fit(housing_numerical_attributes)  
X1 = imputer.transform(housing_numerical_attributes1)


# In[31]:


# Data with replaced NA values.
train_set = pd.DataFrame(X, columns = housing_numerical_attributes.columns, index = housing_numerical_attributes.index)

train_set.insert(9,"ocean_proximity",df["ocean_proximity"],True)
train_set


# In[32]:


# Data with replaced NA values.
test_set = pd.DataFrame(X1, columns = housing_numerical_attributes1.columns, index = housing_numerical_attributes1.index)

test_set.insert(9,"ocean_proximity",df["ocean_proximity"],True)
test_set


# ## Remove target value before normalization

# In[33]:


train_set_without_target = train_set.drop("median_house_value", axis=1) # drop labels for training set
train_set_without_target


# In[34]:


test_set_without_target = test_set.drop("median_house_value", axis=1) # drop labels for training set
test_set_without_target


# ## Normalization

# In[35]:


# Creating pandas series full of zeros to store the standard deviation and the mean from the training set.
std_dev_tr= pd.Series({col:0 for col in train_set_without_target.columns}, dtype="float32")
mean_tr= pd.Series({col:0 for col in train_set_without_target.columns}, dtype="float32")

# Getting the values for the mean and standard deviation from the training dataset.
for col in train_set_without_target.columns:
    std_dev_tr[col]= train_set_without_target[col].std()
    mean_tr[col]= train_set_without_target[col].mean()
    # Changing the training data so it is normalized with the mean and standard deviation from the training set.
    train_set_without_target[col]=(train_set_without_target[col]-mean_tr[col])/std_dev_tr[col]

for col in test_set_without_target.columns:
    # Changing the testing data so it is normalized with the mean and standard deviation from the training set.
    test_set_without_target[col]=(test_set_without_target[col]-mean_tr[col])/std_dev_tr[col]


# In[36]:


train_set_without_target


# In[37]:


train_set_without_target.insert(12,"median_house_value",train_set["median_house_value"]) #Put back target value in train set
train_set_without_target


# In[38]:


train=train_set_without_target
train


# In[39]:


test_set_without_target.insert(12,"median_house_value",test_set["median_house_value"]) #Put back target value in test set
test_set_without_target


# In[40]:


test=test_set_without_target
test


# ## Split between features and target value (labels)

# In[41]:


housing = train.drop("median_house_value", axis=1) # drop labels for training set (FEATURES)
housing_labels = train["median_house_value"].copy() #LABELS FOR TRAIN SET

housing


# In[42]:


housing_labels


# In[43]:


housing_t = test.drop("median_house_value", axis=1) # drop labels for testing set (FEATURES TEST)
housing_labels_t = test["median_house_value"].copy() #LABELS FOR TEST SET

housing_t


# In[44]:


housing_labels_t


# # Ready for modelling

# In[ ]:




