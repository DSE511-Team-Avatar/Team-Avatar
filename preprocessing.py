#                      ....> Checking the data structure <....

def show_structure(df):
    print('\n------------------------------------------------------------------------------------------------------------------------------\n')
    print("Its helpful to see the dataframe structure:\n")
    print(df.head())
    print("\n")
    print('------------------------------------------------------------------------------------------------------------------------------\n')
    
 
def add_more_features(df):
    print('It might be interesting to add the possibly helpful attributes combinations and study their effect on modeling. So:\n')
    df["rooms_per_household"]      = df["total_rooms"]/df["households"]
    df["bedrooms_per_room"]        = df["total_bedrooms"]/df["total_rooms"]
    df["population_per_household"] = df["population"]/df["households"]
    print('Our new dataframe has the following columns:\n')
    print(df.head(n=5))    #....> Show the first 5 rows      


def Handling_categorical(df):
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
