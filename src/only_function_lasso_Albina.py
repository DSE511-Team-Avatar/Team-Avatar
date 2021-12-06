#!/usr/bin/env python
# coding: utf-8

# In[45]:


#DSE511. Project 3. Part3. Modeling. Code for Lasso modeling.Albina Jetybayeva
def lasso_Albina(housing, housing_labels, housing_t, housing_labels_t):
    #Import libraries
    from sklearn import linear_model
    from sklearn.model_selection import GridSearchCV
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
   
    #Lasso
    print("Lasso")
    #%%time
    
    lasso = linear_model.Lasso(alpha=1)
    lasso.fit(housing, housing_labels)
    #%%time
    housing_pred_l = lasso.predict(housing_t)
    
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
    

