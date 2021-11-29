#                      ....> Checking the data structure <....

def show_structure(df):
    print('\n------------------------------------------------------------------------------------------------------------------------------\n')
    print("Its helpful to see the dataframe structure:\n")
    print(df.head())
    print("\n")
    print('Notice: All features are numerical, except the ocean_proximity. Its type is object and its text feature (Categorical feature)')
    print('------------------------------------------------------------------------------------------------------------------------------\n')
    
 
def add_more_features(df):
    print('It might be interesting to add the possibly helpful attributes combinations and study their effect on modeling. So:\n')
    df["rooms_per_household"]      = df["total_rooms"]/df["households"]
    df["bedrooms_per_room"]        = df["total_bedrooms"]/df["total_rooms"]
    df["population_per_household"] = df["population"]/df["households"]
    print('Our new dataframe has the following columns:\n')
    print(df.head(n=5))    #....> Show the first 5 rows      
