# Team-Avatar
### DSE511: Introduction to Data Science and Computing. 
### California Housing Prices

This is a repo for DSE511 project 3. We will be using the dataset for California Housing Prices: (https://www.kaggle.com/camnugent/california-housing-prices).

#### To clone this repro: https://github.com/DSE511-Team-Avatar/Team-Avatar.git
### Team Members: 

- Albina Jetybayeva
- Isidora Fletcher
- Pragya Kandel
- Amirehsan Ghasemi

### Code and Data


The Dataset can be found in the data directory (housing.csv)

To operate our code it is neccessary to use main branch and run the main.py file. Inside it, the two datasets can be used: capped prices (df_Capped) and without capped prices (df_NoCapped). Seven different models can be run on these datasets (Linear Regression, Lasso, Ridge, ElasticNet, Random Forest, Decision Tree, SVR). The details on methods are in the "Methods" section below.

The working reposoitory for the project is given below:

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
│   ├── Part1_eda           <- Scripts for exploratory data analysis.
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

The new computing technologies have widened the scope of machine learning to a great extent. It’s ability to learn from previous computations and independently adapt to new data is making it popular across various disciplines. Various sectors such as business, bioinformatic, computer engineering, pharmaceuticals, medical, climate change, and statistics are using machine learning models to gather knowledge and predict future events [5]. One of the important sectors that machine learning can be used is on real estates to predict the prices of houses. Buying a new house is always a big decision. It gets affected by various factors such as location, size of house, quality of house, future trading price, school zone etc but prioritizing these factors is tough [4]. What would be more important? Is it the location or quality of the house? Machine learning can be used to ease the process of decision making by forecasting the house prices with maximum accuracy of the market trend and the building model based on historic data set [6]. In this project we are trying to understand the factors contributing to housing prices in California. We hypothesis that the values of the houses will be directly related to the number of total rooms, the number of total bedrooms and median income. And inversely related to the housing median age. We will also be investigating how the price is affected by the location (longitude,
latitude and and ocean proximity), the population and the number of households.

## Data

We will be using the dataset for California Housing Prices: (https://www.kaggle.com/camnugent/california-housingprices).
The data is chosen because it has an understandable list of variables and the optimal size between too small and big. The data contains information on houses from the 1990 California census. The data is not cleaned. Although data is old, it can help to learn the regression techniques. The samples are given as
20641 rows and 10 columns of raw data.

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

In this project, we solved a machine learning problem. We had a supervised problem, because we used the house values as labels. The technique that we used to analyze the data was regression. The  process started by doing exploratory data analysis. After the data was loaded, the exploratory data analysis consisted of observing how the features were related. Some examples of tools used for exploratory data analysis were scatter plots, bar plots, etc. These observations drove us to add additional features that were relevant to our problem. The added features were rooms_per_household, bedrooms_per_room and population_per_household. Then, we did the preprocessing of the data, which included handling categorical attributes, dividing the data into testing and training datasets, cleaning and feature scaling the data and separating the target values (median house income). The next step was modeling. The models chosen for this part were Decision Tree, Linear Regression (along with Lasso, Ridge and Elastic Net), Random Forest Regression and SVR. We used the tools available with scikit-learn to fit the data. We used the fit method for each model after creating objects. Then, we evaluated the model’s performance by using the predict method on the test set. Then, we did error evaluation, by using mean squared error (MSE) and taking the square root to obtain the root mean squared error (RMSE). We also used the Mean Absolute Error (MAE) and cross_val_score, scores(where applicable), and wall times(checked in juoyter notebooks for training and predicting). These tools helped us determine how successful the models were and to adjust accordingly. We optimized our model using hyperparameters tuning using “GridSearchCV” from scikit-learn. Finally, we observed and analyzed the importance of each feature using either coefficients of the models (LR, Lasso, Ridge, Elastic Net) or “.feature_importances_” for Random Forest and Decision Tree. The modeling was also tested on two different datasets: “Capped” and “NoCapped”. The capped dataset contained all prices from the original file, while the NoCapped dataset removed the values of prices >$490,000. 

## Conclusion

As most of us want to work in the industry sector once we finish our studies, this type of project equips us with valuable machine learning tools that we can use later on in our professional lives. For example, this project will apply regression models, which is a valuable tool recently learnt in this class. Predicting prices, as mentioned in the introduction, is extremely valuable. This project will help us see which methods are most effective and what information needs to be considered to make a proper prediction. Under-performing models can help us find which parameters are directly affecting the prices and which ones are not. Taking this into consideration will help us determine which combination of parameters will produce the best model. If this study is successful this model can be used for housing prices predictions in other regions. And the structure of the code can be followed to be applied in other projects related to pricing.

## References

[1] Bruno Afonso, Luckeciano Melo, Willian Oliveira, Samuel Sousa, and Lilian Berton. 2019. Housing Prices Prediction with a Deep Learning and Random Forest Ensemble. In Anais do XVI Encontro Nacional de Inteligência Artificial e Computacional (Salvador). SBC, Porto Alegre, RS, Brasil, 389–400. https://doi.org/10.5753/eniac.2019.9300
[2] Okmyung Bin. 2004. A prediction comparison of housing sales prices by parametric versus semi-parametric regressions. Journal of Housing Economics 13, 1 (2004), 68–84. https://doi.org/10.1016/j.jhe.2004.01.001
[3] A. Géron. 2019. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems. O’Reilly Media. https://books.google.com/books?id=HHetDwAAQBAJ
[4] Chaitali Majumder. [n. d.]. House price prediction using machine learning. Retrieved November 5, 2021 from https://nycdatascience.com/blog/studentworks/machine-learning/house-price-prediction-using-machine-learning-2/
[5] Byeonghwa Park and Jae Kwon Bae. 2015. Using machine learning algorithms for housing price prediction: The case of Fairfax County, Virginia housing data. Expert Systems with Applications 42, 6 (2015), 2928–2934. https://doi.org/10.1016/j.eswa.2014.11.040
[6] Subham Sarkar. September 6, 2019. Predicting House prices using classical machine learning and Deep Learning Techniques. Retrieved November 5, 2021 from https://medium.com/analytics-vidhya/predicting-house-prices-using-classical-machine-learning-and-deep-learning-techniques-ad4e55945e2d4
