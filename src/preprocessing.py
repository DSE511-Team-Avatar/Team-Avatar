# DSE511: Introduction to Data Science and Computing - Fall 2021
# California Housing Prices (Preprocessing)
# Team-Avatar

#....> Basic modules
import numpy as np
import pandas as pd

#....> AI/ML modules
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

#******************************************************************************************************************************************
def preprocessing(df):
    #print("Reading the dataframe...\n")
    #df = pd.read_csv('../Data/housing.csv') 
    print("checking shape of data\n")
    rows,columns = df.shape 
    print("Dataframe number of rows: ", rows) 
    print("Dataframe number of columns: ", columns)
    print("Dataframe structure:/n")
    print(df)
    
    #...> combining the attributes
    print('It might be interesting to add the possibly helpful attributes combinations and study their effect on modeling. So:\n')
    df["rooms_per_household"]      = df["total_rooms"]/df["households"]
    df["bedrooms_per_room"]        = df["total_bedrooms"]/df["total_rooms"]
    df["population_per_household"] = df["population"]/df["households"]
    print("New dataframe structure:/n")
    print(df)
    
    #...> handeling categorical attributes i.e. ocean proximity in data frame
    print('------------------------------------------------------------------------------------------------------------------------------\n')
    print('Notice: All features are numerical, except the ocean_proximity. Its type is object and its text feature (Categorical feature)\n')
    print('First, we have to explore this categorical feature\n')
    print(df["ocean_proximity"].value_counts())    #....> Return a Series containing counts of unique values
    print('\nAsssigning numerical values to ocean proximity in the gradient order: the lower the number the further away is the house from the ocean. This gradient is chosen for the better and easier interpretation of models results (feature importances)\n')
    df.loc[df['ocean_proximity'] == 'NEAR OCEAN', 'ocean_proximity'] = 4
    df.loc[df['ocean_proximity'] == 'NEAR BAY', 'ocean_proximity']   = 3
    df.loc[df['ocean_proximity'] == '<1H OCEAN', 'ocean_proximity']  = 2
    df.loc[df['ocean_proximity'] == 'INLAND', 'ocean_proximity']     = 1
    df.loc[df['ocean_proximity'] == 'ISLAND', 'ocean_proximity']     = 0
    print("Dataframe structure:/n")
    print(df)
    
    
    #...> splitting the dataset
    print("Now the data will be splitted in train and test set:\n")
    train_set1, test_set1 = train_test_split(df, test_size=0.2, random_state=1)
    print("Length of Training Data:", len(train_set1))
    print("Length of Testing Data:", len(test_set1))
    
    #...> Checking for missing values
    #..> Training
    print('------------------------------------------------------------------------------------------------------------------------------')
    print('Check if there are any missing values in Training set (NaN or None)\n')
    missing_values_count = train_set1.isnull().sum()
    print("Missing value for Training set: ", missing_values_count[:])
    
    #..> Testing
    print('------------------------------------------------------------------------------------------------------------------------------')
    print('Check if there are any missing values in Testing (NaN or None)\n')
    missing_values_count = test_set1.isnull().sum()
    print("Missing value for Testing set: ", missing_values_count[:])
    
    #...> Show the percentage
    #..> Training
    print('------------------------------------------------------------------------------------------------------------------------------\n')
    print('Find the percentage of the missing values in Training set\n')
    total_cells          = np.product(train_set1.shape)
    missing_values_count = train_set1.isnull().sum()
    total_missing        = missing_values_count.sum()
    percent_missing      = (total_missing/total_cells)*100
    print("percentage of the missing values in Training set: ", percent_missing)
    #..> Testing
    print('Find the percentage of the missing values in Training set\n')
    total_cells          = np.product(test_set1.shape)
    missing_values_count = test_set1.isnull().sum()
    total_missing        = missing_values_count.sum()
    percent_missing      = (total_missing/total_cells)*100
    print("percentage of the missing values in Testing set: ", percent_missing)
    
    #...> Data cleaning
    print('------------------------------------------------------------------------------------------------------------------------------')
    print('Replacing NAN with Imputer (median) using Scikit-learn\n')
    #..> Training set
    imputer = SimpleImputer(strategy = "median") #....> Using medain
    housing_numerical_attributes = train_set1.drop("ocean_proximity", axis = 1)
    imputer.fit(housing_numerical_attributes)  
    X1 = imputer.transform(housing_numerical_attributes)

    #....> Testing Set
    imputer = SimpleImputer(strategy = "median")  #....> Using medain
    housing_numerical_attributes1 = test_set1.drop("ocean_proximity", axis = 1)
    imputer.fit(housing_numerical_attributes)  
    X2 = imputer.transform(housing_numerical_attributes1)
    
    #...> Putting ocean proximity back in train and test set
    #...> Training Set
    train_set = pd.DataFrame(X1, columns = housing_numerical_attributes.columns, index = housing_numerical_attributes.index)
    train_set.insert(9,"ocean_proximity",df["ocean_proximity"],True)
    
    #...> Testing Set
    test_set = pd.DataFrame(X2, columns = housing_numerical_attributes1.columns, index = housing_numerical_attributes1.index)
    test_set.insert(9,"ocean_proximity",df["ocean_proximity"],True)
    print("\nBoth Training and Testing sets the NAN values are replaced with medain\n")
    
    #...> Feature scaling in trainning and testing data set
    train_set_without_target = train_set.drop("median_house_value", axis=1) #....> drop labels for training set    
    test_set_without_target = test_set.drop("median_house_value", axis=1)   #....> drop labels for testing set
    
    #...> Creating pandas series full of zeros to store the standard deviation and the mean from the training set
    std_dev_tr = pd.Series({col:0 for col in train_set_without_target.columns}, dtype="float32")
    mean_tr = pd.Series({col:0 for col in train_set_without_target.columns}, dtype="float32")

    #...> Getting the values for the mean and standard deviation from the training dataset.
    for col in train_set_without_target.columns:
        std_dev_tr[col]= train_set_without_target[col].std()
        mean_tr[col]= train_set_without_target[col].mean()
        #..> Changing the training data so it is normalized with the mean and standard deviation from the training set.
        train_set_without_target[col]=(train_set_without_target[col]-mean_tr[col])/std_dev_tr[col]

    for col in test_set_without_target.columns:
        #..> Changing the testing data so it is normalized with the mean and standard deviation from the training set.
        test_set_without_target[col]=(test_set_without_target[col]-mean_tr[col])/std_dev_tr[col]
     
    train_set_without_target.insert(12,"median_house_value",train_set["median_house_value"]) #Put back the target values for train set
    #...> Relabeling the train data
    train = train_set_without_target
    
    #...> Rebalebeling for test dataset
    test_set_without_target.insert(12,"median_house_value",test_set["median_house_value"]) #Put back the target values for test set
    test = test_set_without_target
    
    #...> splitting between target and feature value in train dataset
    housing = train.drop("median_house_value", axis=1)   #....> drop labels for training set
    housing_labels = train["median_house_value"].copy()
    
    #...> splitting between target and feature value in test dataset
    housing_t = test.drop("median_house_value", axis=1)   #....> drop labels for test set
    housing_labels_t = test["median_house_value"].copy()
    
    print("Congratulations!\n")
    print("Now, you can fit the models to the dataset:\n")
    print("Notice:\n")
    print("X_train = housing\n")
    print("y_train = housing_labels\n")
    print("X_test = housing_t\n")
    print("y_test = housing_labels_t\n")
    
    return housing, housing_labels, housing_t, housing_labels_t 
    
    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
