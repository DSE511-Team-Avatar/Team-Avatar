#!/usr/bin/env python
# coding: utf-8

# In[45]:


#DSE511. Project 3. Part3. Modeling. Code for Linear Regression modeling.Albina Jetybayeva
def linreg_Albina(housing, housing_labels, housing_t, housing_labels_t):
    #Import libraries
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import mean_absolute_error
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import r2_score
    from matplotlib.pyplot import figure
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    
    #%%time
    #Model Linear Regression
    print("Linear Regression")
    lin_reg = LinearRegression()
    lin_reg.fit(housing, housing_labels)
    #%%time
    housing_pred = lin_reg.predict(housing_t)
    
    #Evaluate model
    
    lin_mse = mean_squared_error(housing_labels_t, housing_pred)
    lin_rmse = np.sqrt(lin_mse)
    print("RMSE for Linear Regression: ", lin_rmse)
    
    lin_mae = mean_absolute_error(housing_labels_t, housing_pred)
    print("MAE for Linear Regression: ", lin_mae)
    
    lin_scores = cross_val_score(lin_reg, housing, housing_labels, scoring="neg_mean_squared_error", cv=10)
    lin_rmse_scores = np.sqrt(-lin_scores)
    lin_rmse_scores_mean=lin_rmse_scores.mean()
    print("Cross validation mean score for Linear Regression: ", lin_rmse_scores_mean)
    lr_confidence = lin_reg.score(housing_t, housing_labels_t)
    print("Confidence score for Linear Regression: ", lr_confidence)
    #Visualize the predicted and actual prices
    
    housing_pred_lr = lin_reg.predict(housing_t)
    plt.figure()
    plt.errorbar(housing_labels_t, housing_pred_lr, fmt='o', alpha=0.2)
    plt.title('Linear regression, R2=%.2f' % r2_score(housing_labels_t, housing_pred_lr))
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    importance = lin_reg.coef_
    # summarize feature importance
    for i,v in enumerate(importance):
        print('Feature: %0d, Score: %.5f' % (i,v))
        # plot feature importance
    plt.bar([x for x in range(len(importance))], importance)
    plt.show()
    #To get the exact names of features create the table
    coef_table = pd.DataFrame(list(housing.columns)).copy()
    coef_table.columns = ['Features']
    coef_table.insert(len(coef_table.columns),"Coefs",lin_reg.coef_.transpose())
    print(coef_table)
    coef_table_sorted=coef_table.sort_values(by='Coefs')
    
    figure()
    # Creating a horizontal graph with the values from the pandas Series.
    coef_table_sorted.plot.barh(x='Features', y='Coefs', color="green")
    plt.title("Feature importance for Linear Regression")
    plt.xlabel("Importance")
    plt.ylabel("Features")
    plt.show()
    
    #To evaluate the performance of the Linear Regression model from the different perspective 
    #the only important features observed above will be considered in the next part. 
    #So that "total_rooms", "total_bedrooms", "ocean_proximity", "population_per_household" will be dropped.
    
    #Linear Regression dropped features
    print("Linear Regression with dropped features")
    housing_new = housing.drop(["total_rooms", "total_bedrooms", "ocean_proximity", "population_per_household"], axis=1) # drop labels for training set
    housing_t_new = housing_t.drop(["total_rooms", "total_bedrooms", "ocean_proximity", "population_per_household"], axis=1) # drop labels for training set
    #%%time
    lin_reg1 = LinearRegression()
    lin_reg1.fit(housing_new, housing_labels)
    #%%time
    housing_pred1 = lin_reg1.predict(housing_t_new)
    #Evaluate Model
    lin_mse1 = mean_squared_error(housing_labels_t, housing_pred1)
    lin_rmse1 = np.sqrt(lin_mse1)
    print("RMSE for Linear Regression with dropped features: ", lin_rmse1)
    lin_mae1 = mean_absolute_error(housing_labels_t, housing_pred1)
    print("MAE for Linear Regression with dropped features: ",lin_mae1)
    lin_scores1 = cross_val_score(lin_reg1, housing_new, housing_labels, scoring="neg_mean_squared_error", cv=10)
    lin_rmse_scores1 = np.sqrt(-lin_scores1)
    lin_rmse_scores1_mean=lin_rmse_scores1.mean()
    print("Cross validation mean score for Linear Regression with droppped features: ", lin_rmse_scores1_mean)
    lr_confidence1 = lin_reg1.score(housing_t_new, housing_labels_t)
    print("Confidence score for Linear Regression with dropped features: ", lr_confidence1)
    importance_lr1 = lin_reg1.coef_
    # summarize feature importance
    for i,v in enumerate(importance_lr1):
        print('Feature: %0d, Score: %.5f' % (i,v))
    # plot feature importance
    plt.bar([x for x in range(len(importance_lr1))], importance_lr1)
    plt.show()
    coef_table1 = pd.DataFrame(list(housing_new.columns)).copy()
    coef_table1.columns = ['Features']
    coef_table1.insert(len(coef_table1.columns),"Coefs",lin_reg1.coef_.transpose())
    print(coef_table1)
    coef_table_sorted1=coef_table1.sort_values(by='Coefs')
    figure()
    # Creating a horizontal graph with the values from the pandas Series.
    coef_table_sorted1.plot.barh(x='Features', y='Coefs', color="green")
    plt.title("Feature importance for Linear Regression with dropped features")
    plt.xlabel("Importance")
    plt.ylabel("Features")
    plt.show()
    
    

