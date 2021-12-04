#!/usr/bin/env python
# coding: utf-8

# In[45]:


#DSE511. Project 3. Part3. Modeling. Code for ElasticNet modeling.Albina Jetybayeva
def elasticnet_Albina(housing, housing_labels, housing_t, housing_labels_t):
    #Import libraries
    from sklearn.linear_model import ElasticNet
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import mean_absolute_error
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import r2_score
    from matplotlib.pyplot import figure
   
    
    #ElasticNet
    print("ElasticNet")
    ##%%time
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




