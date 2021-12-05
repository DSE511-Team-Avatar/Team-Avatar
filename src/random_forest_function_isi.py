def isi_random_forest(housing, housing_labels,housing_t, housing_labels_t):
   """Function to implement Random Forest."""
    # Importing RandomForestRegressor for the model.
    from sklearn.ensemble import RandomForestRegressor
    # Importing cross_val_score for error evaluation.
    from sklearn.model_selection import cross_val_score
    # Importing GridSearchCV for tunning.
    from sklearn.model_selection import GridSearchCV
    # Importing sklearn.
    import sklearn
    # Importing what is needed to evaluate the error.
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    
    # Using random forest regressor.
    rf=RandomForestRegressor()
    
    ### Tuning the model. ###
    param_grid = [
     {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
     {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
     ]
    grid_search = GridSearchCV(rf, param_grid, cv=5,
     scoring='neg_mean_squared_error', # We are using MSE
    return_train_score=True)
    grid_search.fit(housing, housing_labels)
    
    # Obtaining the best parameters.
    best_par=grid_search.best_params_
    print(f"The best parameters are: {best_par}")
    
    # Obtaining the best estimator.
    final_model=grid_search.best_estimator_
    print(f"The best esimator is: {final_model}")
    
    # Obtaining the predictions.
    final_predictions=final_model.predict(housing_t)
    
    ### Error evaluation. ###
    # Getting the Mean Absolute Error
    MAE_final= mean_absolute_error(final_predictions,housing_labels_t)
    # Getting the Mean Squared Error
    MSE_final= mean_squared_error(final_predictions,housing_labels_t)
    # Displaying the obtained values for the errors.
    print("MAE is: ", MAE_final, "MSE is: " ,MSE_final)
    # Because the values of MSE are squared, the square root is taken to facilitate the understanding of the value. This way the RMSE is obtained.
    # Root Mean Squared Error
    RMSE_final=np.sqrt(MSE_final)
    print("RMSE is: ", RMSE_final)
                               
    # Fitting the model.
    rf_best_model=final_model.fit(housing, housing_labels)
    # Getting the score for the final model.
    score=rf_best_model.score(housing_t, housing_labels_t)
    print(f"The score for the best model is: {score}")

                               
    # Using cross_val_score with MAE for the data.
    neg_scores=cross_val_score(final_model,housing, housing_labels,scoring="neg_mean_absolute_error", cv=10)
    # Because I was only able to get the negative values using the options above, I added a -.
    final_scores_MAE=(-(neg_scores))
    print("Cross Validation Score.")
    print("scores: ", final_scores_MAE)
    print("scores mean: ", final_scores_MAE.mean())
    print("scores standard deviation: ", final_scores_MAE.std())
                               
    ### Feature Importance. ###
    features=housing.columns
    importances=final_model.feature_importances_
    # Display features and their importances better using pandas.
    combined=pd.Series(importances, features)

    # Importing necessary tools for plotting.
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import figure
        
    print("Plot that shows Feature Importance.")
    figure()
    # Creating a horizontal graph with the values from the pandas Series.
    combined.sort_values().plot.barh(color="blue")
    plt.title("Feature importance for data.")
    plt.xlabel("Importance")
    plt.ylabel("Features")
    plt.show()
    
    
