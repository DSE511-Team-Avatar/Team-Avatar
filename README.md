# Team-Avatar
### DSE511: Introduction to Data Science and Computing (I)
### Final Project: California Housing Prices

This is a repo for DSE511 final project. We will be using the dataset for California Housing Prices: (https://www.kaggle.com/camnugent/california-housing-prices).

#### To clone this repro: https://github.com/DSE511-Team-Avatar/Team-Avatar.git
### Team Members: 

- Albina Jetybayeva
- Isidora Fletcher
- Pragya Kandel
- Amirehsan Ghasemi

### Code and Data


The Dataset can be found in the Data directory (housing.csv)

To operate our code it is neccessary to use main branch and run the main.py file. Inside it, the two datasets can be used: capped prices (df_Capped) and without capped prices (df_NoCapped). Seven different models can be run on these datasets (Linear Regression, Lasso, Ridge, ElasticNet, Random Forest, Decision Tree, SVR). The details on methods are in the "Methods" section below.

The working reposoitory for the project is given below:

```
|
├── Data
│   ├── housing.csv         <- The original data (housing.csv) which will be processed in the python for further use
│
├── Documents
|   └── Preliminary.pdf     <- The file that explains the division of project parts
│   └── Team_Avatar.pdf     <- Preliminary description of the project
|   └── Finalreport.pdf     <- The final report with results and outputs (will be added soon) 
│
├── Jupyter_Notebooks       <- Folder carrying Jupyter notebooks (including Explanatory Data Analysis (EDA) file "Albina_Part1_EDA")
│
│               
│
├── src                     <- Source code (.py) used in this project
│   │
│   ├── main.py             <- Script for running all codes to generate the result
│   |          
│   ├── preprocessing.py    <- Preprocessing and data cleaning
|   |                                       
|   |                      
│   ├── (models: must be called inside main.py)    <- only_function_linreg_Albina.py
|                                                  <- only_function_lasso_Albina.py
|                                                  <- only_function_ridge_Albina.py
|                                                  <- only_function_elasticnet_Albina.py
|                                                  <- isi_rf_function.py
|                                                  <- ehsan_dtr.py
|                                                  <- pragya_svr.py  (will be added soon)
|
| 
├── README.md               <- The README.MD file introduces the project and repository
|
├── requirements.txt        <- Add a requirements.txt file that tracks library that are being used in the project

```
## Introduction

The new computing technologies have widened the scope of machine learning to a great extent. It’s ability to learn from previous computations and independently adapt to new data is making it popular across various disciplines. Various sectors such as business, bioinformatic, computer engineering, pharmaceuticals, medical, climate change, and statistics are using machine learning models to gather knowledge and predict future events [1]. One of the important sectors that machine learning can be used is on real estates to predict the prices of houses. Buying a new house is always a big decision. It gets affected by various factors such as location, size of house, quality of house, future trading price, school zone etc but prioritizing these factors is tough [2]. What would be more important? Is it the location or quality of the house? Machine learning can be used to ease the process of decision making by forecasting the house prices with maximum accuracy of the market trend and the building model based on historic data set [3]. In this project we are trying to understand the factors contributing to housing prices in California. We hypothesize that the values of the houses will be directly related to the ocean proximity, income values, and inversely related to the housing median age and the population in the area (as it will be considered a more private area). We will also be investigating how the price is affected by the location (longitude, latitude and and ocean proximity), the population and the number of households. This study will be formulated as a supervised regression type problem.

## Data

We will be using the dataset for California Housing Prices: (https://www.kaggle.com/camnugent/california-housingprices).
The data is chosen because it has an understandable list of variables and the optimal size between too small and big. The data contains information on houses from the 1990 California census. The data is not cleaned. Although data is old, it can help to learn the regression techniques. The samples are given as
20641 rows and 10 columns of raw data.

- Longitude: A measure of how far west a house is; a higher value is farther west
- Latitude: A measure of how far north a house is; a higher value is farther north
- housing_median_age: Median age of a house within a block; a lower number is a newer building
- total_rooms: Total number of rooms within a block
- total_bedrooms: Total number of bedrooms within a block
- population: Total number of people residing within a block
- households: Total number of households, a group of people residing within a home unit, for a block
- median_income: Median income for households within a block of houses (measured in tens of thousands of US Dollars)
- median_house_value: Median house value for households within a block (measured in US Dollars)
- ocean_proximity: Location of the house w.r.t ocean/sea

All data is numerical, with the exception of ocean_ proximity, which has string input like (“near bay”, “near” ocean”, “inland”, etc.).


## Methods 

In this project, we solved a machine learning problem. We had a supervised problem, because we used the house values as labels. The technique that we used to analyze the data was regression. The  process started by doing exploratory data analysis. After the data was loaded, the exploratory data analysis consisted of observing how the features were related. Some examples of tools used for exploratory data analysis were scatter plots, bar plots, etc. These observations drove us to add additional features that were relevant to our problem. The added features were rooms_per_household, bedrooms_per_room and population_per_household. Then, we did the preprocessing of the data, which included handling categorical attributes, dividing the data into testing and training datasets, cleaning and feature scaling the data and separating the target values (median house income). The next step was modeling. The models chosen for this part were Decision Tree, Linear Regression (along with Lasso, Ridge and Elastic Net), Random Forest Regression and SVR. We used the tools available with scikit-learn to fit the data. We used the fit method for each model after creating objects. Then, we evaluated the model’s performance by using the predict method on the test set. Then, we did error evaluation, by using mean squared error (MSE) and taking the square root to obtain the root mean squared error (RMSE). We also used the Mean Absolute Error (MAE) and cross_val_score, scores(where applicable), and wall times(checked in jupyter notebooks for training and predicting). These tools helped us determine how successful the models were and to adjust accordingly. We optimized our model using hyperparameters tuning using “GridSearchCV” from scikit-learn. Finally, we observed and analyzed the importance of each feature using either coefficients of the models (LR, Lasso, Ridge, Elastic Net) or “.feature_importances_” for Random Forest and Decision Tree. The modeling was also tested on two different datasets: “Capped” and “NoCapped”. The capped dataset contained all prices from the original file, while the NoCapped dataset removed the values of prices >$490,000. 


## Conclusion

As most of us want to work in the industry sector once we finish our studies, this type of project equips us with valuable machine learning tools that we can use later on in our professional lives. For example, in this project we applied regression models, which is a valuable tool. Predicting prices, as mentioned in the introduction, is extremely valuable. The main findings from the project include:
- EDA is an important part of the data science project that helps to make decisions on the data preprocessing, which defines the models performance. In our case, the observation of price distribution helped to notice the “capped” values, the removal of which improved all the models performances significantly. In addition to that, EDA helped us to observe the preliminary feature importances, like strong correlation of prices-income and prices-location.
- These correlations and feature importances were confirmed in most of the models. The higher income and the closer proximity to the ocean implied purchases of more expensive houses.
- Among all the tested models Random Forest  performed the best, as it was expected.
- The best hyperparameters of Random Forest were ‘max_features': 4 and 6 and 'n_estimators': 30 , generating the lowest RMSE among all the models. These values for RMSE were between  $43,000 and $44,000
- Besides ocean proximity/location and income, another important feature was population per household. Then, longitude, latitude and bedroom per room shared around the same importance. The next set of features sharing around the same level of importance were rooms per household and housing median age, And, finally, the last set consisted of total rooms, total bedrooms population and households. The significance of the first few features was expected, since larger households and larger number of bedrooms per room imply the bigger and more expensive properties and thus, higher prices as well. As previously mentioned, EDA showed that people had a preference for the closer to ocean area, which is why the latitude and longitude features are quite significant compared to other features. The remaining features have quite a low importance. This could be because, as we were able to see from EDA and other models, location is the most important feature for people, and the rest of the features end up being less important.


## References

[1] Byeonghwa Park and Jae Kwon Bae. 2015. Using machine learning algorithms for housing price prediction: The case of Fairfax County, Virginia housing data. Expert Systems with Applications 42, 6 (2015), 2928–2934. https://doi.org/10.1016/j.eswa.2014.11.040

[2] Chaitali Majumder. [n. d.]. House price prediction using machine learning. Retrieved November 5, 2021 from https://nycdatascience.com/blog/studentworks/machine-learning/house-price-prediction-using-machine-learning-2/

[3] Subham Sarkar. September 6, 2019. Predicting House prices using classical machine learning and Deep Learning Techniques. Retrieved November 5, 2021 from https://medium.com/analytics-vidhya/predicting-house-prices-using-classical-machine-learning-and-deep-learning-techniques-ad4e55945e2d4

