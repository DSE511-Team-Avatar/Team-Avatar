#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Amirehsan Ghasemi
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import time

#....> AI/ML modules
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

#....> Preventing warnings
import warnings      #....> Prevent from printing the warning of plotting
warnings.simplefilter(action="ignore", category=FutureWarning)
print('Modules are imported\n')  #...> If prints, modules are imported correctly.
print("***************************************************************\n")
#------------------------------------------------------------------------------

def ehsan_dtr(housing, housing_labels, housing_t, housing_labels_t):
    print("Amirehsan Ghasemi, DTR model\n")
    print("Creating a Decision tree regression model:\n")
    dtr = DecisionTreeRegressor()  #...> My model 
    #...................................................................
    print("Then fit to the Training set")
    t0 = time.time()
    dtr.fit(housing, housing_labels)   # Fit to the Training set
    t1 = time.time() 
    print(t1-t0,"seconds Wall Time\n")
    #...................................................................
    print("Report the reults:")
    print("DTR model score Training Set:", dtr.score(housing, housing_labels))
    print("DTR model score Testing Set:", dtr.score(housing_t, housing_labels_t))
    cvs = cross_val_score(dtr, housing, housing_labels, scoring = "neg_mean_squared_error", cv=5)
    cvs_sqrt = np.sqrt(-cvs)
    cvs_mean = np.mean(cvs_sqrt) 
    print("Cross validation score : ", cvs_mean)
    #...................................................................
    print("\nDecision Tree Regression Model Evaluation (on Testing set)")
    t0 = time.time()
    prediction = dtr.predict(housing_t) #...> DTR Model Evaluation (on Testing set)
    t1 = time.time() 
    print(t1-t0,"seconds Wall Time\n")
    #...................................................................
    print("Report the reults:")
    print('MAE for Testing set:', mean_absolute_error(housing_labels_t, prediction))
    print('MSE for Testing set:', mean_squared_error(housing_labels_t, prediction))
    print('RMSE for Testing set:', np.sqrt(mean_squared_error(housing_labels_t, prediction)))
    print("R^2 score for Testing set: ", r2_score(housing_labels_t, prediction))
    #...................................................................
    print("\nSummary of results :\n")
    print("I got 100% score on Training set. On testing set I got almost 63% score\
          because we didn't do any hyperparameters tuning. Due to which depth of \
          tree increased and our model did the overfitting. To solve this problem \
          hyperparameter tuning will be utilized.\n")
    
    #...................................................................
    print("Distribution plot between our label and predicted values:\n")
    figure()
    sns.distplot(housing_labels_t-prediction)
    plt.show()
    print("We can't get any conclusion about our model accrording the bell shape.\
          Bell curve only tell us the range of predicted values are with in the same \
          range as our original data range values are.\n")
    
    print("checking predicted lables and our labeles using a scatter plot:")
    plt.scatter(housing_labels_t, prediction)
    plt.title("predicted lables and our labeles")
    print("Seems the model fits well. We will tune the hyperparameter. To see if we can get better results\n")
     
    #...................................................................
    print("Hyper Parameter is considered.\n")
    dtr = DecisionTreeRegressor()   # Create model: DTR for tuning    
    params = {"criterion": ["squared_error", "absolute_error"],
              "min_samples_split": [10, 20, 40],
              "max_depth": [2, 6, 8],
              "min_samples_leaf": [20, 40, 100],
              "max_leaf_nodes": [5, 20, 100],
              }    
    #....> calculating different regression metrics
    tuning_dtr = GridSearchCV(dtr, param_grid=params, scoring='neg_mean_squared_error', cv=3, verbose=3, n_jobs=-1)
    
    #...................................................................
    print("Fiting the tuned model on the traning set\n")
    print("Please wait. It's tuning the hyperparameters...")
    t0 = time.time()
    tuning_dtr.fit(housing, housing_labels)
    t1 = time.time() 
    print(t1-t0,"seconds Wall Time\n")
    
    #...................................................................
    #           Notice: Uncomment this part if you want the CVS
    #...................................................................
    #cvs_tuned = cross_val_score(tuning_dtr, housing, housing_labels, scoring = "neg_mean_squared_error", cv=5)
    #cvs_tuned_sqrt = np.sqrt(-cvs_tuned)
    #cvs_tuned_mean = np.mean(cvs_tuned_sqrt) 
    #print("-----------------------------------------------------------------------------------------------")
    #print("-----------------------------------------------------------------------------------------------")
    #print("Cross validation score after tuning : ", cvs_tuned_mean)
    
    #...................................................................
    print("Best params: ", tuning_dtr.best_params_)
    print("\n")
    print("Best estimator: ", tuning_dtr.best_estimator_)
    
    #...................................................................
    print("............................................................")
    print("Applying it on the testing set(tuned version)")
    t0 = time.time()
    tuned_pred = tuning_dtr.best_estimator_.predict(housing_t) #....> Applying it on the testing set (tuned version)
    t1 = time.time() 
    print(t1-t0,"seconds Wall Time\n")
    
    #...................................................................
    print("Report the reults:")
    print('MAE for Testing set(tuned):', mean_absolute_error(housing_labels_t,tuned_pred))
    print('MSE for Testing set(tuned):', mean_squared_error(housing_labels_t, tuned_pred))
    print('RMSE for Testing set(tuned):', np.sqrt(mean_squared_error(housing_labels_t, tuned_pred)))
    print("R^2 score for Testing set(tuned): ", r2_score(housing_labels_t,tuned_pred))
    
    #...................................................................
    print("............................................................")
    print("Finiding and plotting feature importances:\n")
    features    = housing_t.columns
    importances = tuning_dtr.best_estimator_.feature_importances_
    combined    = pd.Series(importances, features)
     
    #...> plot:
    figure()
    combined.sort_values().plot.barh(color="b")
    plt.title("Feature importance")
    plt.xlabel("Importance")
    plt.ylabel("Features")
    plt.show()
    
    
    #...................................................................
    print("............................................................")
    print("Plotting the results:\n")
    test = pd.DataFrame({'Predicted':tuned_pred,'Actual':housing_labels_t})
    fig = plt.figure(figsize=(16,8))
    test = test.reset_index()
    test = test.drop(['index'],axis=1)
    plt.plot(test[:50])
    plt.legend(['Actual','Predicted'])
    sns.jointplot(x='Actual',y='Predicted',data=test,kind="reg")
    
    
    
    
    
    
    
