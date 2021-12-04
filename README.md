# Team-Avatar
### DSE511: Introduction to Data Science and Computing. 
### California Housing Prices

This is a repo for DSE511 project 3. We will be using the dataset for California Housing Prices: (https://www.kaggle.com/camnugent/california-housing-prices).

### Team Members: 

- Albina Jetybayeva
- Isidora Fletcher
- Pragya Kandel
- Amirehsan Ghasemi

### Code and Data

The working reposoitory for the project is given below:
Dataset can be found in the data directory (will be updated soon): 

```
├── README.md               <- The README.MD file introduces the project and repository
├── Data
│   ├── housing.csv         <- The original data (housing.csv) which will be processed in the python for further use. 
│
├── Documents
│   ├── Team_Avatar.pdf     <- Preliminary description of the project
│   └── Preliminary.pdf     <- The file that explains the division of project parts
|   └── Finalreport.pdf     <- The final report with results and outputs 
│
├── Jupyter_Notebooks       <- Folder carrying Jupyter notebooks.
│
├── requirements.txt        <- Add a requirements.txt file that tracks library that are being used in the project.
│               
│
├── src                     <- Source code used in this project
│   │
│   ├── Part1_data          <- Import and split data in trainning and testing set
│   ├── Part1_eda           <- Scripts for explanatory data analysis.
│   ├── Part2_preprocessing |---data_clean    <- Cleaning the data: work on NAN, null, missing values, etc
|   |                       |---text_hand     <- Handling text and Categorical attributes 
|   |                       |---Feat_scaling  <- Feature scaling and normalization  
|   |                      
│   └──Part3_modeling       |--- Models       <- linear regression, Randomforest, SVM, lasso/elastic net
|   |                       |---Optimization  <- Hyperarameter tuning
|   |                       |---Error_analysis <- Using cross validation and mean square error
|   |                       |---Feat_extract   <- Find out importance of each attributes
│
└── main.py                 <- Script for running all codes to generate the result

```
## Introduction

In this project we are trying to understand the factors contributing to housing prices specifically in California.  The chosen dataset on California was used for the machine learning basics introduction in the book by Aurélien Géron ’Hands-On Machine learning with Scikit-Learn and TensorFlow’ [1]. We hypothesize that the values of the houses will be directly related to the ocean proximity, income values, and inversely related to the housing median age and the population in the area (as it will be considered a more private area). We will also be investigating how the price is affected by the location (longitude, latitude and and ocean proximity), the population, the number of households etc. This study will be formulated as a supervised regression type problem, and the regression modelling will be used with such models as Decision Tree, Linear Regression (LR), Random Forest Regression, and Support Vector Regression (SVR) to predict the housing prices based on the combination of the various variables available in the dataset.


## Data

For our investigation we used the dataset for California Housing Prices, which contains information on houses from the 1990 California census [7]. Although the information from this dataset was old, it was helpful to learn the regression techniques. Moreover, another reason why this data was chosen was because it had an understandable list of variables and the optimal size for ML practice. The raw dataset was not cleaned and it consisted of 20641 rows and 10 columns. 

Longitude - A measure of how far west a house is; a higher value is farther west
Latitude - A measure of how far north a house is; a higher value is farther north
housing_median_age - Median age of a house within a block; a lower number is a newer building
total_rooms - Total number of rooms within a block
total_bedrooms - Total number of bedrooms within a block
population - Total number of people residing within a block
households - Total number of households, a group of people residing within a home unit, for a block
median_income - Median income for households within a block of houses (measured in tens of thousands of US Dollars)
median_house_value - Median house value for households within a block (measured in US Dollars)
ocean_proximity - Location of the house w.r.t ocean/sea

All data is numerical, with the exception of ocean_ proximity, which has string input like (“near bay”, “near” ocean”, “inland”, etc.).

## Methods

In this project, we solved a machine learning problem. We had a supervised problem, since we used the house prices values as labels. The technique that we applied to analyze the data was regression. After loading the raw data, the process started by doing exploratory data analysis (EDA). The EDA consisted of observing how the features were related initially. Generally, based on the EDA, it was clear that cleaning the missing values, features scaling and categorical feature handling are required on the data. It is important  to mention that in order to avoid data leakage and bias we split the data (80% train and 20% test sets) before cleaning and feature scaling. The next step was modeling. The models chosen for this part were Decision Tree, LR (along with Lasso, Ridge and Elastic Net), Random Forest Regression and SVR. Then, we evaluated the models’ performance by using the predict method on the test set. After, we did error evaluation, by using root mean squared error (RMSE), Mean Absolute Error (MAE), model score [2], wall time (train and predict), and cross validation score. These tools helped us determine how successful the models were and to adjust accordingly. We optimized our models using hyperparameters tuning “GridSearchCV” from scikit-learn. Finally, we observed and analyzed the importance of each feature using either coefficients of the models (LR, Lasso, Ridge, Elastic Net) or “.feature_importances_” for Random Forest . The modeling was also tested on two different datasets: “capped” and “uncapped”. The capped dataset contained all prices from the original file, while the uncapped dataset removed the values of prices >$490,000. The details on this decision will be provided in the EDA sections. 


## Conclusion

EDA is an important part of the data science project that helps to make decisions on the data preprocessing, which defines the models performance. In our case, the observation of price distribution helped to notice the “capped” values, the removal of which improved all the models performances significantly. In addition to that, EDA helped us to observe the preliminary feature importances, like strong correlation of prices-income and prices-location.
- These correlations and feature importances were confirmed in most of the models. The higher income and the closer proximity to ocean implied purchases of more expensive houses.
- Among all the tested models Random Forest  performed the best, as it was expected.
- The best hyperparameters of Random Forest were ‘max_features': 4, 'n_estimators': 30 along with the lowest among all the models RMSE $43,686.
- Besides ocean proximity/location and income, the other important features included population per household, bedroom per room. These were expected, as larger households and larger number of bedrooms per room imply the bigger and more expensive properties and thus, higher prices as well.
This project helped us see which methods are most effective and what information needs to be considered to make a proper prediction. Under-performing models helped us find which parameters needed to be changed. Taking this into consideration helped us determine which combination of parameters will produce the best model. This type of  model can be used for housing prices predictions in other regions. And the structure of the code can be followed to be applied in other projects related to pricing.
The future work of improvement for this project will include the possible application of Principle Component Analysis (PCA), Non-negative Matrix Factorization (NMF), to reduce the dimensionality of the problem, which might improve the models performance in both terms of error reduction and faster estimation. Moreover, other models like GaussianProcesses, Ensemble learning models can be tested and tuned more to get even better results. Finally, for the future work and easier representation of categorical feature “ocean proximity” it might be helpful to create one binary attribute per each category: one attribute would equal to 1 when the category is true (and 0 otherwise).


## References

[1] Kaggle. California Housing Priceshttps://www.kaggle.com/camnugent/california-housing-prices

[2] Scikit-Learn Library. Linear Regression https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html

