import numpy as np
import pandas as pd

# read in all our data
hp_data = pd.read_csv('Data/housing.csv')

# set seed for reproducibility
np.random.seed(0)


# Check if there are any missing values (NaN or None)
# look at the first five rows of the hp_data file.
hp_data.head() 

