#!/usr/bin/env python
# coding: utf-8

# In[45]:


#DSE511. Project 3. Part3. Modeling. Code for Linear Regression, Lasso, Ridge, ElasticNet modeling.Albina Jetybayeva
def regression_Albina(housing, housing_labels, housing_t, housing_labels_t):
    #Import basic libraries
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import LabelEncoder
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import train_test_split
    import matplotlib as mpl
    #%%time
    #Model Linear Regression
    print("Linear Regression")
    from sklearn.linear_model import LinearRegression
    lin_reg = LinearRegression()
    lin_reg.fit(housing, housing_labels)
    #%%time
    housing_pred = lin_reg.predict(housing_t)
    #Evaluate model
    from sklearn.metrics import mean_squared_error
    lin_mse = mean_squared_error(housing_labels_t, housing_pred)
    lin_rmse = np.sqrt(lin_mse)
    print("RMSE for Linear Regression: ", lin_rmse)
    from sklearn.metrics import mean_absolute_error
    lin_mae = mean_absolute_error(housing_labels_t, housing_pred)
    print("MAE for Linear Regression: ", lin_mae)
    from sklearn.model_selection import cross_val_score
    lin_scores = cross_val_score(lin_reg, housing, housing_labels, scoring="neg_mean_squared_error", cv=10)
    lin_rmse_scores = np.sqrt(-lin_scores)
    lin_rmse_scores_mean=lin_rmse_scores.mean()
    print("Cross validation mean score for Linear Regression: ", lin_rmse_scores_mean)
    lr_confidence = lin_reg.score(housing_t, housing_labels_t)
    print("Confidence score for Linear Regression: ", lr_confidence)
    #Visualize the predicted and actual prices
    from sklearn.metrics import r2_score
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
    from matplotlib.pyplot import figure
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
    
    #Ridge
    print("Ridge")
    #%%time
    from sklearn.linear_model import Ridge
    ridge = Ridge(alpha=1)
    ridge.fit(housing, housing_labels)
    #%%time
    housing_pred_r = ridge.predict(housing_t)
    from sklearn.model_selection import GridSearchCV
    #Grid Search
    clf = Ridge()
    grid_values = {'alpha':[1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]}
    grid_clf_acc = GridSearchCV(clf, param_grid = grid_values, scoring='neg_mean_squared_error',cv=5)
    grid_clf_acc.fit(housing, housing_labels)
    print("Best tuned alpha: ", grid_clf_acc.best_estimator_)
    #Predict values based on new parameters
    y_pred_acc = grid_clf_acc.predict(housing_t)
    #Evaluate model
    lin_mse_r = mean_squared_error(housing_labels_t, housing_pred_r)
    lin_rmse_r = np.sqrt(lin_mse_r)
    print("RMSE for Ridge: ", lin_rmse_r)
    lin_mse_r1 = mean_squared_error(housing_labels_t, y_pred_acc) #alpha=10
    lin_rmse_r1 = np.sqrt(lin_mse_r1)
    print("RMSE for Ridge - tuned hyperparameter: ", lin_rmse_r1)
    lin_mae_r = mean_absolute_error(housing_labels_t, housing_pred_r)
    print("MAE for Ridge: ", lin_mae_r)
    lin_mae_r1 = mean_absolute_error(housing_labels_t, y_pred_acc) #alpha=10
    print("MAE for Ridge - tuned hyperparameter: ", lin_mae_r1)
    lin_scores_r = cross_val_score(ridge, housing, housing_labels, scoring="neg_mean_squared_error", cv=10)
    lin_rmse_scores_r = np.sqrt(-lin_scores_r)
    lin_rmse_scores_r_mean=lin_rmse_scores_r.mean()
    print("Cross validation mean score for Ridge: ", lin_rmse_scores_r_mean)
    lr_confidence_r = ridge.score(housing_t, housing_labels_t)
    print("Confidence score for Ridge: ", lr_confidence_r)
    importance_r1 = ridge.coef_
    # summarize feature importance
    for i,v in enumerate(importance_r1):
        print('Feature: %0d, Score: %.5f' % (i,v))
        # plot feature importance
    plt.bar([x for x in range(len(importance_r1))], importance_r1)
    plt.show()
    coef_table_r = pd.DataFrame(list(housing.columns)).copy()
    coef_table_r.columns = ['Features']
    coef_table_r.insert(len(coef_table_r.columns),"Coefs",ridge.coef_.transpose())
    print(coef_table_r)
    coef_table_sorted_r=coef_table_r.sort_values(by='Coefs')
    figure()
    # Creating a horizontal graph with the values from the pandas Series.
    coef_table_sorted_r.plot.barh(x='Features', y='Coefs', color="green")
    plt.title("Feature importance for Ridge")
    plt.xlabel("Importance")
    plt.ylabel("Features")
    plt.show()
    
    #Lasso
    print("Lasso")
    #%%time
    from sklearn import linear_model
    lasso = linear_model.Lasso(alpha=1)
    lasso.fit(housing, housing_labels)
    #%%time
    housing_pred_l = lasso.predict(housing_t)
    from sklearn.model_selection import GridSearchCV
    #Grid Search
    clf2 = linear_model.Lasso()
    grid_values2 = {'alpha':[1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]}
    grid_clf_acc2 = GridSearchCV(clf2, param_grid = grid_values2, scoring='neg_mean_squared_error',cv=5)
    grid_clf_acc2.fit(housing, housing_labels)
    print("Best tuned alpha: ", grid_clf_acc2.best_estimator_)
    #Predict values based on new parameters
    y_pred_acc2 = grid_clf_acc2.predict(housing_t)
    #Evaluate model
    lin_mse_l = mean_squared_error(housing_labels_t, housing_pred_l)
    lin_rmse_l = np.sqrt(lin_mse_l)
    print("RMSE for Lasso: ", lin_rmse_l)
    lin_mse_l2 = mean_squared_error(housing_labels_t, y_pred_acc2)
    lin_rmse_l2 = np.sqrt(lin_mse_l2)
    print("RMSE for Lasso - tuned hyperparameter: ", lin_rmse_l2)
    lin_mae_l = mean_absolute_error(housing_labels_t, housing_pred_l)
    print("MAE for Lasso: ", lin_mae_l)
    lin_mae_l2 = mean_absolute_error(housing_labels_t, y_pred_acc2)
    print("MAE for Lasso - tuned hyperparameter: ", lin_mae_l2)
    lin_scores_l = cross_val_score(lasso, housing, housing_labels, scoring="neg_mean_squared_error", cv=10)
    lin_rmse_scores_l = np.sqrt(-lin_scores_l)
    lin_rmse_scores_l_mean=lin_rmse_scores_l.mean()
    print("Cross validation mean score for Lasso: ", lin_rmse_scores_l_mean)
    lin_scores_l2 = cross_val_score(grid_clf_acc2, housing, housing_labels, scoring="neg_mean_squared_error", cv=10)
    lin_rmse_scores_l2 = np.sqrt(-lin_scores_l2)
    lin_rmse_scores_l2_mean=lin_rmse_scores_l2.mean()
    print("Cross validation mean score for Lasso - tuned hyperparameter: ", lin_rmse_scores_l2_mean)
    lr_confidence_l = lasso.score(housing_t, housing_labels_t)
    print("Confidence score for Lasso: ", lr_confidence_l)
    lasso2 = linear_model.Lasso(alpha=100)
    lasso2.fit(housing, housing_labels)
    lr_confidence_l2 = lasso2.score(housing_t, housing_labels_t)
    print("Confidence score for Lasso - tuned hyperparameter: ", lr_confidence_l2)
    importance_las = lasso.coef_
    # summarize feature importance
    for i,v in enumerate(importance_las):
        print('Feature: %0d, Score: %.5f' % (i,v))
    # plot feature importance
    plt.bar([x for x in range(len(importance_las))], importance_las)
    plt.show()
    coef_table_l = pd.DataFrame(list(housing.columns)).copy()
    coef_table_l.columns = ['Features']
    coef_table_l.insert(len(coef_table_l.columns),"Coefs",lasso.coef_.transpose())
    print(coef_table_l)
    coef_table_sorted_l=coef_table_l.sort_values(by='Coefs')
    figure()
    # Creating a horizontal graph with the values from the pandas Series.
    coef_table_sorted_l.plot.barh(x='Features', y='Coefs', color="green")
    plt.title("Feature importance for Lasso")
    plt.xlabel("Importance")
    plt.ylabel("Features")
    plt.show()
    
    #ElasticNet
    print("ElasticNet")
    ##%%time
    from sklearn.linear_model import ElasticNet
    en = ElasticNet(alpha=1) #l1_ratio=0.5
    en.fit(housing, housing_labels)
    ##%%time
    housing_pred_en = en.predict(housing_t)
    #Grid Search
    clf3 = ElasticNet()
    grid_values3 = {'alpha':[1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100], 'l1_ratio': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]}
    grid_clf_acc3 = GridSearchCV(clf3, param_grid = grid_values3, scoring='neg_mean_squared_error',cv=5)
    grid_clf_acc3.fit(housing, housing_labels)
    print("Best tuned alpha and l1_ratio: ", grid_clf_acc3.best_estimator_)
    #Predict values based on new parameters
    y_pred_acc3 = grid_clf_acc3.predict(housing_t)
    #Evaluating model
    lin_mse_en = mean_squared_error(housing_labels_t, housing_pred_en)
    lin_rmse_en = np.sqrt(lin_mse_en)
    print("RMSE for ElasticNet: ", lin_rmse_en) #l1_ratio=0.5, alpha =1
    en1 = ElasticNet(alpha=100) #l1_ratio=0.5
    en1.fit(housing, housing_labels)
    housing_pred_en1 = en1.predict(housing_t)
    lin_mse_en1 = mean_squared_error(housing_labels_t, housing_pred_en1)
    lin_rmse_en1 = np.sqrt(lin_mse_en1)
    print("RMSE for ElasticNet - tuned hyperparameter: ", lin_rmse_en1) #l1_ratio=0.5, alpha=100
    lin_mae_en = mean_absolute_error(housing_labels_t, housing_pred_en)
    print("MAE for ElasticNet: ", lin_mae_en)
    lin_scores_en = cross_val_score(en, housing, housing_labels, scoring="neg_mean_squared_error", cv=10)
    lin_rmse_scores_en = np.sqrt(-lin_scores_en)
    lin_rmse_scores_en_mean=lin_rmse_scores_en.mean()
    print("Cross validation mean score for ElasticNet: ", lin_rmse_scores_en_mean)    
    lr_confidence_en = en.score(housing_t, housing_labels_t)
    print("Confidence score for ElasticNet: ", lr_confidence_en)
    coef_table_en = pd.DataFrame(list(housing.columns)).copy()
    coef_table_en.columns = ['Features']
    coef_table_en.insert(len(coef_table_en.columns),"Coefs",en.coef_.transpose())
    print(coef_table_en)
    importance_en = en.coef_
    # summarize feature importance
    for i,v in enumerate(importance_en):
        print('Feature: %0d, Score: %.5f' % (i,v))
    # plot feature importance
    plt.bar([x for x in range(len(importance_en))], importance_en)
    plt.show()
    coef_table_sorted_en=coef_table_en.sort_values(by='Coefs')
    figure()
    # Creating a horizontal graph with the values from the pandas Series.
    coef_table_sorted_en.plot.barh(x='Features', y='Coefs', color="green")
    plt.title("Feature importance for ElasticNet")
    plt.xlabel("Importance")
    plt.ylabel("Features")
    plt.show()


# In[ ]:




