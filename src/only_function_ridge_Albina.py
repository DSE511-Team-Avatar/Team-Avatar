#!/usr/bin/env python
# coding: utf-8

# In[45]:


#DSE511. Project 3. Part3. Modeling. Code for Ridge modeling.Albina Jetybayeva
def ridge_Albina(housing, housing_labels, housing_t, housing_labels_t):
    #Import basic libraries
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import mean_absolute_error
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import r2_score
    from matplotlib.pyplot import figure
    
   
    
    #Ridge
    print("Ridge")
    #%%time
    ridge = Ridge(alpha=1)
    ridge.fit(housing, housing_labels)
    #%%time
    housing_pred_r = ridge.predict(housing_t)

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

