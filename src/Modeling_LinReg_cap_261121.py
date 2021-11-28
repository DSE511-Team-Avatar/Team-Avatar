#!/usr/bin/env python
# coding: utf-8

# # DSE511. Project 3. Part3. Modeling. 
# ## Code for Linear Regression modeling.
# ### Albina Jetybayeva

# In this project we are trying to understand the factors contributing to housing prices in California. We hypothesis that the values of the houses will be directly related to the number of total rooms, the number of total bedrooms and median income. And inversely related to the housing median age. We will also be investigating how the price is affected by the location (longitude, latitude and and ocean proximity), the population and the number of households.
# 
# In this part, the cleaned, preprocessed dataset will be modeled using the simplest Linear Regerssion as a baseline. The Linear Regression will then be analyzed based on the importance of each feature and then tested with the extraction of only the most contributing features. After that the regression models like Lasso, Ridge and Elastic Net, which help to decrease the model complexity, that is the number of predictors, will be also tested. Each model will be discussed separately and the results will be compared.
# 
# These metrics will be used to assess the perfromance of the models:
# - mean_squared_error
# - mean_absolute_error
# - cross_val_score
# - Confidence, model.score()
# - Coefficients, model.coef_

# ## Data import and preprocessing

# In[156]:


#Importing the base libraries

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import matplotlib as mpl
mpl.style.use(['ggplot']) #use ggplot style


# In[157]:


# Getting the raw data
df = pd.read_csv('housing.csv') # Notice: Raw data is in the Data folder
df


# In[158]:


print('There are {} rows and {} columns in train'.format(df.shape[0],df.shape[1]))


# In[159]:


# As it was dsicussed in Part 1. Explanatory Data Analysis, it might be interesting to add the possibly helpful 
#attributes combinations and study their effect on modeling too

df["rooms_per_household"] = df["total_rooms"]/df["households"]
df["bedrooms_per_room"] = df["total_bedrooms"]/df["total_rooms"]
df["population_per_household"]=df["population"]/df["households"]
df


# ## Handling categorical attributes
# 
# As in this dataset there is present one categorical attribute "ocean proximity", for each category the specific number will be assigned. The categories will be changed as follows: NEAR OCEAN = 4; NEAR BAY = 3; <1H OCEAN = 2; INLAND = 1; ISLAND =0. So there will be 5 categories (0-4 numbers) in total.

# In[160]:


# Asssigning numerical values to ocean proximity in the gradient order: the lower the number the further away is the house from the ocean
# this gradient is chosen for the better and easier interpretation of models results (feature importances)

df.loc[df['ocean_proximity'] == 'NEAR OCEAN', 'ocean_proximity'] = 4
df.loc[df['ocean_proximity'] == 'NEAR BAY', 'ocean_proximity'] = 3
df.loc[df['ocean_proximity'] == '<1H OCEAN', 'ocean_proximity'] = 2
df.loc[df['ocean_proximity'] == 'INLAND', 'ocean_proximity'] = 1
df.loc[df['ocean_proximity'] == 'ISLAND', 'ocean_proximity'] = 0

df


# ## Data split
# 
# Now the data will be splitted in train and test set to avoid the data leakage and bias during the further preprocessing steps, which include cleaning the missing values and feature scaling.

# In[161]:


# Splitting the data into training and testing sets.
train_set1, test_set1 = train_test_split(df, test_size=0.2, random_state=1)
print("Training Data", len(train_set1))
print("Testing Data", len(test_set1))


# ## Data cleaning on train and test
# 
# Changing the missing values with the median on a train set.

# In[162]:


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


# Changing the missing values with the median (from a train set) on a test set.

# In[163]:


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


# In[164]:


# Data with replaced NAN values.
# Put back the ocean proximity in train set
train_set = pd.DataFrame(X, columns = housing_numerical_attributes.columns, index = housing_numerical_attributes.index)

train_set.insert(9,"ocean_proximity",df["ocean_proximity"],True)
train_set


# In[165]:


# Data with replaced NAN values.
# Put back the ocean proximity in test set
test_set = pd.DataFrame(X1, columns = housing_numerical_attributes1.columns, index = housing_numerical_attributes1.index)

test_set.insert(9,"ocean_proximity",df["ocean_proximity"],True)
test_set


# ## Feature Scaling
# 
# As the attributes have very different ranges, it is recommended ro do the normalization on them, since Machine Learning algorithms don’t perform well when the input numerical attributes have very different scales. 
# 
# Standardization will be applied, which first subtracts the mean value (so standardized values always have a zero mean), and then it divides by the standard deviation so that the resulting distribution has unit variance. The advantage of standardization is that is is much less affected by outliers.
# 
# As scaling the target values is generally not required, these will not be scaled. So first the target labels will be dropped and then added after the nscaling.

# In[166]:


train_set_without_target = train_set.drop("median_house_value", axis=1) # drop labels for training set 
train_set_without_target


# In[167]:


test_set_without_target = test_set.drop("median_house_value", axis=1) # drop labels for test set
test_set_without_target


# In[168]:


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


# In[169]:


train_set_without_target


# In[170]:


train_set_without_target.insert(12,"median_house_value",train_set["median_house_value"]) #Put back the target values for train set
train_set_without_target


# In[171]:


train=train_set_without_target
train


# In[172]:


test_set_without_target.insert(12,"median_house_value",test_set["median_house_value"]) #Put back the target values for test set
test_set_without_target


# In[173]:


test=test_set_without_target
test


# ## Modeling Linear Regression

# In[174]:


# First, we will split between features and target value (labels) for train set

housing = train.drop("median_house_value", axis=1) # drop labels for training set
housing_labels = train["median_house_value"].copy()

housing


# In[175]:


housing_labels


# In[176]:


# Second, we will split between features and target value (labels) for test set

housing_t = test.drop("median_house_value", axis=1) # drop labels for test set
housing_labels_t = test["median_house_value"].copy()

housing_t


# In[177]:


housing_labels_t


# LinearRegression fits a linear model with coefficients w = (w1, …, wp) to minimize the residual sum of squares between the observed targets in the dataset, and the targets predicted by the linear approximation.

# In[178]:


get_ipython().run_cell_magic('time', '', '\nfrom sklearn.linear_model import LinearRegression\n\nlin_reg = LinearRegression()\nlin_reg.fit(housing, housing_labels)')


# In[179]:


get_ipython().run_cell_magic('time', '', 'housing_pred = lin_reg.predict(housing_t)')


# First metrics that will be used to asses the performance of the model will be the mean_squared_error. This function computes mean square error, a risk metric corresponding to the expected value of the squared (quadratic) error or loss (https://scikit-learn.org/stable/modules/model_evaluation.html#mean-squared-error). In statistics, the mean squared error (MSE)[1] or mean squared deviation (MSD) of an estimator (of a procedure for estimating an unobserved quantity) measures the average of the squares of the errors—that is, the average squared difference between the estimated values and the actual value. 

# In[180]:


from sklearn.metrics import mean_squared_error

lin_mse = mean_squared_error(housing_labels_t, housing_pred)
lin_rmse = np.sqrt(lin_mse)
lin_rmse


# It is clear that that the error is quite large. Considering that the price ranges between USD 120,000 and USD 265,000, the typical prediction error of USD 68,628 is too big. This is an example of a model underfitting the training data. When this happens it might be a result of the features not providing enough information to make good predictions, or that the model is not powerful
# enough. The main ways to fix underfitting are to select a more powerful model, to feed the training algorithm with better features, or to reduce the constraints on the model. Linear Regression model is not regularized, so this rules out the last option. However, playing with features (removing or adding some) can be done later.
# 
# another metrics that can be used is the mean absolute error. In statistics, mean absolute error (MAE) is a measure of errors between paired observations expressing the same phenomenon. It is thus an arithmetic average of the absolute errors |e(i)|=|y(i)-x(i)|, where y(i) is the prediction and x(i) is the true value. 

# In[181]:


from sklearn.metrics import mean_absolute_error

lin_mae = mean_absolute_error(housing_labels_t, housing_pred)
lin_mae


# As it can be seen, MAE is also quite high for this linear regression model.
# 
# Another helpful metrics used for the model evaluation, especially for trees models, is the cross validation score. This function randomly splits the training set into 10 distinct subsets called folds, then it trains and evaluates the model 10 times, picking a different fold for evaluation every time and training on the other 9 folds. The result is an array containing the 10 evaluation scores. Scikit-Learn’s cross-validation features expect a utility function (greater is better) rather than a cost function (lower is better), so the scoring function is actually the opposite of the MSE (i.e., a negative value). Cross-validation allows to get not only an estimate of the performance of your model, but also a measure of how precise this estimate is (i.e., its standard deviation).

# In[182]:


from sklearn.model_selection import cross_val_score

lin_scores = cross_val_score(lin_reg, housing, housing_labels, scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)


# In[183]:


def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

display_scores(lin_rmse_scores)


# Another metrics for the model performance is the model score. The coefficient of determination R^2 is defined as (1-u/v), where u is the residual sum of squares ((y_true - y_pred)** 2).sum() and v is the total sum of squares ((y_true - y_true.mean()) ** 2).sum(). The best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse). 

# In[184]:


lr_confidence = lin_reg.score(housing_t, housing_labels_t)
print("lr confidence: ", lr_confidence)


# For this model the score is 0.63, which is not high as expected from the other previous metrics.

# In[185]:


#Visualize the predicted and actual prices
from sklearn.metrics import r2_score

housing_pred_lr = lin_reg.predict(housing_t)
plt.figure()
plt.errorbar(housing_labels_t, housing_pred_lr, fmt='o', alpha=0.2)
plt.title('Linear regression, R2=%.2f' % r2_score(housing_labels_t, housing_pred_lr))
plt.xlabel('Actual')
plt.ylabel('Predicted')


# As it was seen the values of prices were capped with USD 500,000 median_house_value. The capped house value may be a problem for a precise modeling since it is the target attribute (labels). The Machine Learning algorithms may learn that prices never go beyond that limit. To check how the model will perform without these capped prices those values will be removed from the training set (and also from the test set, since the system should not be evaluated poorly if it predicts values beyond USD 500,000) and tested in a separate code.
# 
# To evaluate the features importance of linear regression model, the coefficients of the model will be extracted. Regression coefficients are estimates of the unknown population parameters and describe the relationship between a predictor variable and the response.The sign of each coefficient indicates the direction of the relationship between a predictor variable and the response variable. A positive sign indicates that as the predictor variable increases, the response variable also increases.
# A negative sign indicates that as the predictor variable increases, the response variable decreases. The coefficient value represents the mean change in the response given a one unit change in the predictor. So the larger the coefficient can be interpreted as more weight and siginifcance is for this feature.

# In[186]:


import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.style.use(['ggplot']) #use ggplot style

importance = lin_reg.coef_
# summarize feature importance
for i,v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
plt.bar([x for x in range(len(importance))], importance)
plt.show()


# In[187]:


#To get the exact names of features create the table
coef_table = pd.DataFrame(list(housing.columns)).copy()
coef_table.columns = ['Features']
coef_table.insert(len(coef_table.columns),"Coefs",lin_reg.coef_.transpose())
coef_table


# In[188]:


coef_table_sorted=coef_table.sort_values(by='Coefs')


# In[189]:


from matplotlib.pyplot import figure
figure()
# Creating a horizontal graph with the values from the pandas Series.

coef_table_sorted.plot.barh(x='Features', y='Coefs', color="green")
plt.title("Feature importance for uncapped values.")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.show()


# 1) It can be seen that among all the features the income has the highest positive signifcance, followed by the households snd bedrooms per room numbers. These are expected, as higher salaries correlate with buying more expensive houses, while larger households and larger amount of bedrooms per rooms imply the larger properties and thus, higher prices as well.
# 
# 2) Interestingly, the strong negative coefficients are noticed for the longtitude, lattitude and the price. By ananlyzing and visualizing that, it can be seen that the more north and the east the direction on the map the cheaper the houses are. By looking at the map of Califronia from part1 explanatory data analysis, it can be seen that this direction represents exactly the moving more into continental part of the state and away from teh coastal line. So this interetsing finsing highlights that.
# 
# 3) Another strong negative realtionship was observed for the population, which can be explained by the fact that the lower the population in the area the higher the price, as it can be considered a more private area.
# 
# 4) Interestingly, the small positive correlation was found for the house age and price, which contradicts with the expected result. As the newer houses were predicted to be more expensive. Most probably, those houses which are older, are placed in the favorbale and popular locations, and location as we observed plays an important role in defining the house price and this effect overpasses the age factor.
# 
# 5) Although the location is important it might be difficult to interpret the ocean proximity in this dataset as they are assigned as different numbers. One issue with this representation is that ML algorithms will assume that two nearby values are more similar than two distant values. This may be fine in some cases (e.g., for ordered categories such as “bad”, “average”, “good”, “excellent”), this was done for ocean proximity as well by asssigning numerical values to ocean proximity in the gradient order: the lower the number the further away is the house from the ocean. This gradient is chosen for the better and easier interpretation of models results (feature importances). Still it can be seen that the small positive coefficient indicates that with closer proximity to ocean increases the housing prices. For the future work and easier representation what can be done is to use a common solution to create one binary attribute per category: one attribute equal to 1 when the category is “<1H OCEAN” (and 0 otherwise), another attribute equal to 1 when the category is “INLAND” (and 0 otherwise), and so on.This can be studied further.

# ## Extract only important features and run again LinReg
# 
# To evaluate the performance of the Linear Regression model from the different perspective the only important features observed above will be considered in the next part. So that "total_rooms", "total_bedrooms", "ocean_proximity", "population_per_household" will be dropped.

# In[190]:


housing_new = housing.drop(["total_rooms", "total_bedrooms", "ocean_proximity", "population_per_household"], axis=1) # drop labels for training set
housing_new


# In[191]:


housing_t_new = housing_t.drop(["total_rooms", "total_bedrooms", "ocean_proximity", "population_per_household"], axis=1) # drop labels for training set
housing_t_new


# In[192]:


get_ipython().run_cell_magic('time', '', 'lin_reg1 = LinearRegression()\nlin_reg1.fit(housing_new, housing_labels)')


# In[193]:


get_ipython().run_cell_magic('time', '', 'housing_pred1 = lin_reg1.predict(housing_t_new)')


# In[194]:


lin_mse1 = mean_squared_error(housing_labels_t, housing_pred1)
lin_rmse1 = np.sqrt(lin_mse1)
lin_rmse1


# Evaluating the RMSE and comparing with the previous value, it can be seen that the difference is not large. It is very slightly smaller, but this is not very impactful. So other strategies should be applied to reduce RMSE.

# In[195]:


lin_mae1 = mean_absolute_error(housing_labels_t, housing_pred1)
lin_mae1


# Same for the MAE, validation score, confidence score values, where the difference is insignificant.

# In[196]:


lin_scores1 = cross_val_score(lin_reg1, housing_new, housing_labels, scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores1 = np.sqrt(-lin_scores1)
display_scores(lin_rmse_scores1)


# In[197]:


lr_confidence1 = lin_reg1.score(housing_t_new, housing_labels_t)
print("lr confidence: ", lr_confidence1)


# In[198]:


importance_lr1 = lin_reg1.coef_
# summarize feature importance
for i,v in enumerate(importance_lr1):
    print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
plt.bar([x for x in range(len(importance_lr1))], importance_lr1)
plt.show()


# In[199]:


coef_table1 = pd.DataFrame(list(housing_new.columns)).copy()
coef_table1.columns = ['Features']
coef_table1.insert(len(coef_table1.columns),"Coefs",lin_reg1.coef_.transpose())
coef_table1


# In[200]:


coef_table_sorted1=coef_table1.sort_values(by='Coefs')


# In[201]:


figure()
# Creating a horizontal graph with the values from the pandas Series.

coef_table_sorted1.plot.barh(x='Features', y='Coefs', color="green")
plt.title("Feature importance for uncapped values.")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.show()


# Generally, the coefficients stayed the same as in the previous case, and the perfromance of the model was not improved significantly.

# ## Lasso, Ridge or Elastic Net
# 
# As the results were not improved much, the other related regression models woul be tested like Lasso, Ridge and Elastic Net and their optimized results will be compared with the Linear Regression.

# ## Ridge

# This model solves a regression model where the loss function is the linear least squares function and regularization is given by the l2-norm. Also known as Ridge Regression or Tikhonov regularization. This estimator has built-in support for multi-variate regression (i.e., when y is a 2d-array of shape (n_samples, n_targets)). In Ridge Regression, the OLS loss function is augmented in such a way that we not only minimize the sum of squared residuals but also penalize the size of parameter estimates, in order to shrink them towards zero.
# 
# L2 regularization (Ridge) - better for dense data.

# In[202]:


get_ipython().run_cell_magic('time', '', '\nfrom sklearn.linear_model import Ridge\nridge = Ridge(alpha=1)\nridge.fit(housing, housing_labels)')


# In[203]:


get_ipython().run_cell_magic('time', '', 'housing_pred_r = ridge.predict(housing_t)')


# Ridge has an alpha parameter that can be optimized and grisdearch will be used for that. Alpha represents the regularization strength; must be a positive float. Regularization improves the conditioning of the problem and reduces the variance of the estimates. Larger values specify stronger regularization. 

# In[204]:


from sklearn.model_selection import GridSearchCV

#Grid Search
clf = Ridge()
grid_values = {'alpha':[1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]}
grid_clf_acc = GridSearchCV(clf, param_grid = grid_values, scoring='neg_mean_squared_error',cv=5)
grid_clf_acc.fit(housing, housing_labels)

print(grid_clf_acc.best_estimator_)

#Predict values based on new parameters
y_pred_acc = grid_clf_acc.predict(housing_t)


# It can be seen that the results are similar to Linear Regression model for MSE (alpha=1).

# In[205]:


lin_mse_r = mean_squared_error(housing_labels_t, housing_pred_r)
lin_rmse_r = np.sqrt(lin_mse_r)
lin_rmse_r


# For optimized alpha=10, RMSE also doesnt show much improvement

# In[206]:


lin_mse_r1 = mean_squared_error(housing_labels_t, y_pred_acc) #alpha=10
lin_rmse_r1 = np.sqrt(lin_mse_r1)
lin_rmse_r1


# The case is the same for MAE

# In[207]:


lin_mae_r = mean_absolute_error(housing_labels_t, housing_pred_r)
lin_mae_r


# In[208]:


lin_mae_r1 = mean_absolute_error(housing_labels_t, y_pred_acc) #alpha=10
lin_mae_r1


# Looking at the other metrics, again no large improvements are noticed compared to baseline Linear Regression.

# In[209]:


lin_scores_r = cross_val_score(ridge, housing, housing_labels, scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores_r = np.sqrt(-lin_scores_r)
display_scores(lin_rmse_scores_r)


# In[210]:


lr_confidence_r = ridge.score(housing_t, housing_labels_t)
print("lr confidence: ", lr_confidence_r)


# In[211]:


importance_r1 = ridge.coef_
# summarize feature importance
for i,v in enumerate(importance_r1):
    print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
plt.bar([x for x in range(len(importance_r1))], importance_r1)
plt.show()


# In[212]:


coef_table_r = pd.DataFrame(list(housing.columns)).copy()
coef_table_r.columns = ['Features']
coef_table_r.insert(len(coef_table_r.columns),"Coefs",ridge.coef_.transpose())
coef_table_r


# In[213]:


coef_table_sorted_r=coef_table_r.sort_values(by='Coefs')


# In[214]:


figure()
# Creating a horizontal graph with the values from the pandas Series.

coef_table_sorted_r.plot.barh(x='Features', y='Coefs', color="green")
plt.title("Feature importance for uncapped values.")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.show()


# Generally, the coefficients stayed the same as in the Linear Regression case, and the perfromance of the model was not improved significantly.

# ## Lasso

# Lasso, or Least Absolute Shrinkage and Selection Operator, is quite similar conceptually to ridge regression. It also adds a penalty for non-zero coefficients, but unlike ridge regression which penalizes sum of squared coefficients (the so-called L2 penalty), lasso penalizes the sum of their absolute values (L1 penalty). As a result, for high values of λ, many coefficients are exactly zeroed under lasso, which is never the case in ridge regression.
# 
# Linear Model trained with L1 prior as regularizer (aka the Lasso).
# 
# L1 regularization (Lasso) - better for sparse data.

# In[215]:


get_ipython().run_cell_magic('time', '', 'from sklearn import linear_model\nlasso = linear_model.Lasso(alpha=1)\nlasso.fit(housing, housing_labels)')


# In[216]:


get_ipython().run_cell_magic('time', '', 'housing_pred_l = lasso.predict(housing_t)')


# Similarly to Ridge, Lasso also has an alpha parameter that can be optimized and grisdearch will be used for that. Alpha is a constant that multiplies the L1 term. Defaults to 1.0. alpha = 0 is equivalent to an ordinary least square, solved by the LinearRegression object. 

# In[217]:


from sklearn.model_selection import GridSearchCV

#Grid Search
clf2 = linear_model.Lasso()
grid_values2 = {'alpha':[1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]}
grid_clf_acc2 = GridSearchCV(clf2, param_grid = grid_values2, scoring='neg_mean_squared_error',cv=5)
grid_clf_acc2.fit(housing, housing_labels)

print(grid_clf_acc2.best_estimator_)

#Predict values based on new parameters
y_pred_acc2 = grid_clf_acc2.predict(housing_t)


# Again, checking all the same metrics for Lasso, no large differences are observed for the baseline Linear Regression and Lasso for both alpha=1 and optimized alpha=100.

# In[218]:


lin_mse_l = mean_squared_error(housing_labels_t, housing_pred_l)
lin_rmse_l = np.sqrt(lin_mse_l)
lin_rmse_l


# In[219]:


lin_mse_l2 = mean_squared_error(housing_labels_t, y_pred_acc2)
lin_rmse_l2 = np.sqrt(lin_mse_l2)
lin_rmse_l2


# In[220]:


lin_mae_l = mean_absolute_error(housing_labels_t, housing_pred_l)
lin_mae_l


# In[221]:


lin_mae_l2 = mean_absolute_error(housing_labels_t, y_pred_acc2)
lin_mae_l2


# In[222]:


lin_scores_l = cross_val_score(lasso, housing, housing_labels, scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores_l = np.sqrt(-lin_scores_l)
display_scores(lin_rmse_scores_l)


# In[223]:


lin_scores_l2 = cross_val_score(grid_clf_acc2, housing, housing_labels, scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores_l2 = np.sqrt(-lin_scores_l2)
display_scores(lin_rmse_scores_l2)


# In[224]:


lr_confidence_l = lasso.score(housing_t, housing_labels_t)
print("lr confidence: ", lr_confidence_l)


# In[225]:


lasso2 = linear_model.Lasso(alpha=100)
lasso2.fit(housing, housing_labels)
lr_confidence_l2 = lasso2.score(housing_t, housing_labels_t)
print("lr confidence: ", lr_confidence_l2)


# In[226]:


importance_las = lasso.coef_
# summarize feature importance
for i,v in enumerate(importance_las):
    print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
plt.bar([x for x in range(len(importance_las))], importance_las)
plt.show()


# In[227]:


coef_table_l = pd.DataFrame(list(housing.columns)).copy()
coef_table_l.columns = ['Features']
coef_table_l.insert(len(coef_table_l.columns),"Coefs",lasso.coef_.transpose())
coef_table_l


# In[228]:


coef_table_sorted_l=coef_table_l.sort_values(by='Coefs')


# In[229]:


figure()
# Creating a horizontal graph with the values from the pandas Series.

coef_table_sorted_l.plot.barh(x='Features', y='Coefs', color="green")
plt.title("Feature importance for uncapped values.")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.show()


# ## Elastic Net

# Elastic Net first emerged as a result of critique on lasso, whose variable selection can be too dependent on data and thus unstable. The solution is to combine the penalties of ridge regression and lasso to get the best of both worlds.
# 
# So it is a linear regression with combined L1 and L2 priors as regularizer.

# In[230]:


get_ipython().run_cell_magic('time', '', 'from sklearn.linear_model import ElasticNet\nen = ElasticNet(alpha=1) #l1_ratio=0.5\nen.fit(housing, housing_labels)')


# In[231]:


get_ipython().run_cell_magic('time', '', 'housing_pred_en = en.predict(housing_t)')


# There are two parameters that can be optimized: l1_ratio and alpha. Alpha is a constant that multiplies the penalty terms. Defaults to 1.0. See the notes for the exact mathematical meaning of this parameter. alpha = 0 is equivalent to an ordinary least square, solved by the LinearRegression object. The ElasticNet mixing parameter, with 0 <= l1_ratio <= 1. For l1_ratio = 0 the penalty is an L2 penalty. For l1_ratio = 1 it is an L1 penalty. For 0 < l1_ratio < 1, the penalty is a combination of L1 and L2.

# In[232]:


#Grid Search
clf3 = ElasticNet()
grid_values3 = {'alpha':[1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100], 'l1_ratio': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]}
grid_clf_acc3 = GridSearchCV(clf3, param_grid = grid_values3, scoring='neg_mean_squared_error',cv=5)
grid_clf_acc3.fit(housing, housing_labels)

print(grid_clf_acc3.best_estimator_)

#Predict values based on new parameters
y_pred_acc3 = grid_clf_acc3.predict(housing_t)


# Checking the RMSE for Elastic Net with alpha=1 and l1_ratio=0.5, it can be seen that the erros is increased for this case compared to baseline Linear Regression, meaning its performing even worse than LinReg.
# 
# Moreover, using the optimized parameter alpha=100 and l1_ratio =0.5 (not 1, as found by the gridsearch, otherwise assigning to 1 will run the same model as Lasso previously, that has already been tested). The results of these parameters are even worse, meaning that the model with these parameters is not the optimal for this task.

# In[233]:


lin_mse_en = mean_squared_error(housing_labels_t, housing_pred_en)
lin_rmse_en = np.sqrt(lin_mse_en)
lin_rmse_en #l1_ratio=0.5, alpha =1


# In[234]:


en1 = ElasticNet(alpha=100) #l1_ratio=0.5
en1.fit(housing, housing_labels)
housing_pred_en1 = en1.predict(housing_t)
lin_mse_en1 = mean_squared_error(housing_labels_t, housing_pred_en1)
lin_rmse_en1 = np.sqrt(lin_mse_en1)
lin_rmse_en1 #l1_ratio=0.5


# Checking the rest of the parameters, it can be observed that the performance of ElasticNet model is siginifcantly lower than the baseline Linear Regression.

# In[235]:


lin_mae_en = mean_absolute_error(housing_labels_t, housing_pred_en)
lin_mae_en


# In[236]:


lin_scores_en = cross_val_score(en, housing, housing_labels, scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores_en = np.sqrt(-lin_scores_en)
display_scores(lin_rmse_scores_en)


# In[237]:


lr_confidence_en = en.score(housing_t, housing_labels_t)
print("lr confidence: ", lr_confidence_en)


# In[238]:


coef_table_en = pd.DataFrame(list(housing.columns)).copy()
coef_table_en.columns = ['Features']
coef_table_en.insert(len(coef_table_en.columns),"Coefs",en.coef_.transpose())
coef_table_en


# In[239]:


importance_en = en.coef_
# summarize feature importance
for i,v in enumerate(importance_en):
    print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
plt.bar([x for x in range(len(importance_en))], importance_en)
plt.show()


# In[240]:


coef_table_sorted_en=coef_table_en.sort_values(by='Coefs')


# In[241]:


figure()
# Creating a horizontal graph with the values from the pandas Series.

coef_table_sorted_en.plot.barh(x='Features', y='Coefs', color="green")
plt.title("Feature importance for uncapped values.")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.show()


# The changes in features coefficients can be seen in this case. Where all of them had the decreased absolute values. However, as this model performs worse than the baseline Linear Regression, it would not be recommended to use this less reliable model.

# In[ ]:




