#!/usr/bin/env python
# coding: utf-8

# # Project 3. Part1. Explanatory Data Analysis. Team Avatar 
# Albina Jetybayeva
# 
# In this project we are trying to understand the factors contributing to housing prices in California. We hypothesis that the values of the houses will be directly related to the number of total rooms, the number of total bedrooms and median income. And inversely related to the housing median age. We will also be investigating how the price is affected by the location (longitude, latitude and and ocean proximity), the population and the number of households.
# 
# This is a supervised regression type problem and the dataset is suitable for the regression modelling and quantity prediction.
# 
# We will be using the dataset for California Housing Prices: (https://www.kaggle.com/camnugent/california-housing-
# prices).
# 
# * Longitude -  A measure of how far west a house is; a higher value is farther west
# * Latitude  -  A measure of how far north a house is; a higher value is farther north
# * housing_median_age - Median age of a house within a block; a lower number is a newer building
# * total_rooms - Total number of rooms within a block
# * total_bedrooms - Total number of bedrooms within a block
# * population - Total number of people residing within a block
# * households - Total number of households, a group of people residing within a home unit, for a block
# * median_income - Median income for households within a block of houses (measured in tens of thousands of US Dollars)
# * median_house_value - Median house value for households within a block (measured in US Dollars)
# * ocean_proximity - Location of the house w.r.t ocean/sea
# 
# In this part, the dataset will be preliminarily studied to observe the nature and features of the data and what preprocessing and analysis can be done on it. As the explanatory data analysis is the initial part of the project, it would be submitted in the Jupyter notebook format. The main observations of the data analysis will be highlighted after the code and infographics.

# In[33]:


#Import the basic important libraries
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.style.use(['ggplot']) #use ggplot style


# In[34]:


#Extract data from the file:
data = pd.read_csv('housing.csv')
data.head()


# In[35]:


print('There are {} rows and {} columns in train'.format(data.shape[0],data.shape[1]))


# In[36]:


#Split the data first and do all feature transformations after the test_train splitting on the train set only to avoid data leakage

from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(data, test_size=0.2, random_state=1)

print("Training Data", len(train_set))
print("Testing Data", len(test_set))


# In[37]:


#Use info to get a quick description of the data, in particular the total number of rows, and each attribute’s type and number of non-null values
data.info()


# There are 20,640 total cases in the dataset, which is fairly small for Machine Learning standards, but it’s good for introductory level study. It's important to notice that the total_bed rooms attribute has only 20,433 non-null values, meaning that 207 cases are missing this feature. This will need to be considered during data preprocessing.
# 
# All features are numerical, except the ocean_proximity. Its type is object, and by loaded the data from a CSV file we saw that it is a text attribute. Looking at the top five rows, it is probably noticed that the values in the ocean_proximity column are repetitive, which means that it is a categorical attribute. To see what categories exist and how many cases belong to each category value_counts() method can be used.

# In[38]:


data["ocean_proximity"].value_counts()


# In[39]:


#To see visaully it can be also represented in a histogram:
freq = data.ocean_proximity.value_counts()
plt.figure(figsize=(10, 6))
plt.bar(freq.index, height = freq,ec='#21209c',color='#008891')
plt.xlabel('Ocean Proximity', fontsize=16)
plt.ylabel('No. of Households', fontsize=16)
plt.show()


# It can be seen that most of the houses are in <1H ocean (less than 1 hour drive to ocean) and Inland category.
# To observe the summary of the numerical attributes describe() method will be used.

# In[40]:


data.describe()


# The count, mean, min, and max rows are self-explanatory. The std row shows the standard deviation, which measures how dispersed the values are. The 25%, 50%, and 75% rows show the corresponding percentiles: a percentile indicates the value below which a given percentage of observations in a group of observations falls.
# 
# Another quick way to get a feel of the type of data we are dealing with is to plot a histogram for each numerical attribute. A histogram shows the number of instances (on the vertical axis) that have a given value range (on the horizontal axis).

# In[41]:


import matplotlib.pyplot as plt
data.hist(bins=50, figsize=(20,15))
plt.show()


# Important observations from the  histograms:
#  
# * From the code main source description it was found that the housing median age and the median house value were capped. That is why there are many houses with USD 500,000 median_house_value and 50 years house_median_age. The capped house value may be a  problem for a precise modeling since it is the target attribute (labels). The Machine Learning algorithms may learn that prices never go beyond that limit. It is neccessary to check with the client team (the team that will use system’s output) to see if this is a problem or not. If they tell that they need precise predictions even beyond USD 500,000, then it would be required to either:
# 1) Collect proper labels for the districts whose labels were capped.
# 2) Remove those districts from the training set (and also from the test set, since the system should not be evaluated poorly if it predicts values beyond USD 500,000).
# * All the attributes have very different scales. Thus, feature scaling would be required as feature scaling is essential for machine learning algorithms that calculate distances between data. If not scale, the feature with a higher value range starts dominating when calculating distances.
# *  Finally, many histograms are tail heavy: they extend much farther to the right of the median than to the left. This may make it a bit harder for some Machine Learning algorithms to detect patterns. Thus, these might be also transformed to have more bell-shaped distributions.

# Since there is geographical information (latitude and longitude), it is a good idea to create a scatterplot of all districts to visualize the data and housing locations.

# In[42]:


data.plot(kind="scatter", x="longitude", y="latitude")


# This looks like California coastal side, but other than that it is hard to see any other particular pattern. Changing the transparency of dots (houses) by setting the alpha option to 0.1 makes it much easier to visualize the places where there is a high density of data points.

# In[43]:


data.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)


# As this feature helps to see the high-density areas, it is clear that the highest density appears in the regions namely San Jose, the Bay Area, around Los Angeles, and San Diego. Moreover, there is a long line of fairly high density in the Central Valley, in particular around Sacramento and Fresno.

# Now it is useful to do the integration of the information on housing prices. The radius of each circle represents the district’s population (option s), and the color represents the price (option c).

# In[44]:


data.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4, s=data["population"]/100, label="population", figsize=(10,7),
c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True)
plt.legend()


# This image shows that the housing prices are very much related to the location (e.g., close to the ocean) and to the population density, as was expected.
# 
# By looking at California map it can be seen that the regions of the most expensive houses are San Francisco,Sacramento, Los Angeles and San Diego areas. These are the major cities in California, so it is understandable.
# 

# In[45]:


import folium

world_map = folium.Map(location=[37, -119], zoom_start=6,tiles='OpenStreetMap')

world_map


# To visualize better the mapping of house prices in California, the overlaying will be used on Califronia map.

# In[46]:


data.plot(kind='scatter', x='longitude', y='latitude', alpha=0.4, s=data['population']/100, label='population', figsize=(10,7),c='median_house_value', cmap=plt.get_cmap('jet'), colorbar=True)

import matplotlib.image as mpimg
california_img=mpimg.imread('cali.png')
plt.imshow(california_img, extent=[-124.55, -113.80, 32.45, 42.05], alpha=0.5)
plt.ylabel("Latitude", fontsize=14)
plt.xlabel("Longitude", fontsize=14)
plt.legend() 
plt.show()


# It is clear again, that the houses closer to the ocean are more expensive as well as those in the regions of major cities as LA, San Francisco and San Diego.
# 
# Since the dataset is medium-size, the standard correlation coefficient (also called Pearson’s r) between every pair of attributes can be checked using the corr() method.
# Specifically it is useful to see how much each attribute correlates with the median house value.

# In[47]:


corr_matrix = data.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)


# The correlation coefficient ranges from –1 to 1. When it is close to 1, it means that there is a strong positive correlation; for example, as the median house value tends to incerase when the median income goes up. This is expected, as more income allows people to buy more expensive houses. When the coefficient is close to –1, it means that there is a strong negative correlation; a small negative correlation i snoticed between the latitude and the median house value (i.e., prices have a slight tendency to decrease going north). This might be explained by the fact that going more north means being in more inland area, further away from ocean (and we have seen that ocean proximity plays an important role in prices determination). Moreover, going north might also mean colder temperatures, which might be less preferrable for people, thus, the prices reduce as demand is less there. Finally, coefficients close to zero mean that there is no linear correlation. Therefore, this needs more investigation.
# 
# It is also useful to try out various attribute combinations. For example, the
# total number of rooms in a district is not very helpful if you don’t know how many households there are. What is really useful is the number of rooms per household. Similarly, the total number of bedrooms by itself is not very helpful: it might be good to compare it to the number of rooms. And the population per household also seems like an interesting attribute combination to look at.

# In[48]:


data["rooms_per_household"] = data["total_rooms"]/data["households"]
data["bedrooms_per_room"] = data["total_bedrooms"]/data["total_rooms"]
data["population_per_household"]=data["population"]/data["households"]

corr_matrix = data.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)


# It is interesting to notice that the new bedrooms_per_room attribute is much more correlated with the median house value than the total number of rooms or bedrooms. Apparently houses with a lower bedroom/room ratio tend to be more expensive. The number of rooms per household is also more informative than the total number of rooms in a district—obviously the larger the houses, the more expensive they are.
# 
# 
# To better visualize it, seaborn heatmap can be used to view correlations between old and new features in dataset.

# In[49]:


import seaborn as sns 


corr1 = data.corr()
plt.figure(figsize=(8,6))
pltheatmap =sns.heatmap(corr1, annot=True, annot_kws={"size": 8})


# Another way to check for correlation between attributes is to use Pandas’scatter_matrix function, which plots every numerical attribute against every other numerical attribute and a histogram of each attribute along the main diagonal. Since there are now 12 numerical attributes, we would get 144 plots, which would not fit on a page, so let’s just focus on a few promising attributes that seem most correlated with the median housing value.

# In[50]:


from pandas.plotting import scatter_matrix
attributes = ["median_house_value", "median_income", "rooms_per_household", "bedrooms_per_room", "housing_median_age"]
scatter_matrix(data[attributes], figsize=(14, 14))


# The most promising attribute to predict the median house value is the median
# income, so let’s zoom in on their correlation scatterplot.

# In[51]:


data.plot(kind="scatter", x="median_income", y="median_house_value",
alpha=0.1)


# This plot reveals a few things. First, the correlation is indeed very strong;  the upward trend is clearly seen and the points are not too dispersed. Second, the price cap that we noticed earlier is clearly visible as a horizontal line at USD 500,000.
# 
# ALso it is worth to try to zoom into bedroom_per_room attribute.

# In[52]:


data.plot(kind="scatter", x="bedrooms_per_room", y="median_house_value",
             alpha=0.2)
plt.axis([0.05, 1.1, 0, 520000])
plt.show()


# It can be seen that there is slightly obvious negative correltaion between price and bedrooms_per_room. So it might be a case that more expensive houses have more rooms for other purposes like offices, areas for entertainmnet (cinema, video games, etc.), gyms, studios, and others.So the ratio of bedroom and other rooms is lower in these more expensive houses.

# As it was observed above, the income and ocean proximity are probably most important parameters defining the house prices. So to visualize their relations, seaborn pairplot of median income vs median house value with hue as Ocean Proximity will be done.

# In[53]:


sns.pairplot(data, height=4, aspect=2, vars=["median_income","median_house_value"], hue="ocean_proximity")


# Analyzing the bottom left graph, it can be seen that "inland" houses mostly lie in the region of lower income and house prices, as expected. Most of the "near bay" and "near ocean" houses are placed closer to the higher income-price region, while "<1 hour drive to ocean" more dispersed in the middle. These clearly show the main trends, athough there are obviously expections and other important parameters as discussed, such as bedrooms per rooms and rooms per household, as well as other attributes not present in the dataset, which need careful consideration for more accurate prediction.

# # Summary
# 
# It was observed that several parameters such as income and ocean proximity play an important role in house prices determination. Other imortant parameters such as bedrooms per rooms, rooms per household, age, latitude, although do not have the clear linear correlation, still might be useful for more accurate modelling. These will be investigated further.
# 
# This part is useful to start off on the right foot and quickly gain the insights about the data that will help to get a first reasonably good prototype. This is an iterative process: once we get a prototype up and running, we can analyze its output to gain more insights and come back to this exploration step.
