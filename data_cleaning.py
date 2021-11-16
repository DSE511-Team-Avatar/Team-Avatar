# File created by Isi with part of Ehsan's code to make merging easier.
# Observing missing values
missing_values_count = df.isnull().sum()
missing_values_count[:]

total_cells   = np.product(df.shape)
total_missing = missing_values_count.sum()
percent_missing = (total_missing/total_cells)*100
print('Percent of data that is missing:', percent_missing)

imputer = SimpleImputer(strategy = "median")
housing_numerical_attributes = df.drop("ocean_proximity", axis = 1) # We need to see how to put this back.
imputer.fit(housing_numerical_attributes)  
X = imputer.transform(housing_numerical_attributes)

# Data with replaced NA values.
# I changed the name from new_df to housing.
housing = pd.DataFrame(X, columns = housing_numerical_attributes.columns, index = housing_numerical_attributes.index)
