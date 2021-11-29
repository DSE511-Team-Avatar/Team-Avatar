import numpy as np



#....> Checking the data structure 
def show_structure(df):
    print('\n------------------------------------------------------------------------------------------------------------------------------\n')
    print("Its helpful to see the dataframe structure:\n")
    print(df)
    print("\n")
    print('------------------------------------------------------------------------------------------------------------------------------\n')
    
 
def adding_additional_features(df):
    print('It might be interesting to add the possibly helpful attributes combinations and study their effect on modeling. So:\n')
    df["rooms_per_household"]      = df["total_rooms"]/df["households"]
    df["bedrooms_per_room"]        = df["total_bedrooms"]/df["total_rooms"]
    df["population_per_household"] = df["population"]/df["households"]
    return df     


def handling_categorical_attributes(df):
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
    return df

def data_splitting(df):
    from sklearn.model_selection import train_test_split
    print("Now the data will be splitted in train and test set.\n")
    train_set1, test_set1 = train_test_split(df, test_size=0.2, random_state=1)
    print("Training Data:", len(train_set1))
    print("Testing Data:", len(test_set1))      
    return train_set1, test_set1     
    

def check_null(dataset):
    print('------------------------------------------------------------------------------------------------------------------------------')
    print('Check if there are any missing values (NaN or None)\n')
    missing_values_count = dataset.isnull().sum()
    print(missing_values_count[:])
    


def missing_percentage(dataset):
    print('------------------------------------------------------------------------------------------------------------------------------')
    print('Find the percentage of the missing values in our dataset\n')
    total_cells          = np.product(dataset.shape)
    missing_values_count = dataset.isnull().sum()
    total_missing        = missing_values_count.sum()
    percent_missing      = (total_missing/total_cells)*100
    print(percent_missing)
    

def data_cleaning(train_set, test_set):
    print('------------------------------------------------------------------------------------------------------------------------------')
    print('Replacing NAN with Imputer (median) using Scikit-learn\n')
    from sklearn.impute import SimpleImputer
    #....> Training Set
    imputer = SimpleImputer(strategy = "median") #....> Using medain
    training_numerical_attributes = train_set.drop("ocean_proximity", axis = 1)
    imputer.fit(training_numerical_attributes)  
    X1 = imputer.transform(training_numerical_attributes)
    #....> Testing Set
    imputer = SimpleImputer(strategy = "median")  #....> Using medain
    testing_numerical_attributes = test_set.drop("ocean_proximity", axis = 1)
    imputer.fit(testing_numerical_attributes)  
    X2 = imputer.transform(testing_numerical_attributes)
    print("Both Training and Testing sets the NAN values are replaced with medain")
    return X1, X2