# Creating pandas series full of zeros to store the standard deviation and the mean.
# This is just a section of the code, because I need other pre-processing parts to make it work. Complete code in jupyter notebooks. 
std_dev= pd.Series({col:0 for col in housing.columns}, dtype="float32")
mean= pd.Series({col:0 for col in housing.columns}, dtype="float32")

# Getting the values for the mean and standard deviation from the data.
for col in housing.columns:
    std_dev[col]= housing[col].std()
    mean[col]= housing[col].mean()
    # Changing the data so it is normalized with the mean and standard deviation.
    housing[col]=(housing[col]-mean[col])/std_dev[col]
    
# Now housing is the data normalized, but it is missing the ocean_proximity column.