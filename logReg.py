import numpy as np
from sklearn import linear_model, datasets
from sklearn.feature_selection import SelectFromModel

#-------------------------------------------------------------------------
'''
    Big Data Analytics - Project 1
'''

#-------------------------------------------------------------------------
# EXAMPLE DATA AND ANALYSIS 

# # import some data to play with
# iris = datasets.load_iris()
# X = iris.data[:, :2]  # we only take the first two features.
# Y = iris.target

# # print ("iris: ", iris)
# print("X shape: ", X.shape)
# # print ("Y:", Y)

# h = .02  # step size in the mesh

# ridgeReg = linear_model.Ridge (alpha = .5) 
# logreg = linear_model.LogisticRegression(C=1e5)
# linreg = linear_model.LinearRegression()

# # Fit data for ridge, logistic and linear regression 
# ridgeReg.fit(X, Y)
# print("Coefficient matrix for ridge:", ridgeReg.coef_)
# logreg.fit(X, Y)
# print("Coefficient matrix for logit:", logreg.coef_)
# linreg.fit(X, Y)
# print("Coefficient matrix linear:", linreg.coef_)

# # Select most important features from each 
# modelRidge = SelectFromModel(ridgeReg, prefit=True)
# X_new_ridge = modelRidge.transform(X)
# print("New X ridge shape: ", X_new_ridge.shape)
# #print("New X: ", X_new_ridge)

# modelLog = SelectFromModel(logreg, prefit=True)
# X_new_log = modelLog.transform(X)
# print("New X log shape: ", X_new_log.shape)
# #print("New X: ", X_new_log)

# modelLin = SelectFromModel(linreg, prefit=True)
# X_new_lin = modelLin.transform(X)
# print("New X linear shape: ", X_new_lin.shape)
# #print("New X: ", X_new_lin)



#-------------------------------------------------------------------------

# Samples: classified tweets (polarity from -1 to 1)
# Features: weather features such as humidity, pressure, temperature, wind speed
xtest = load() 		# Matrix of shape[n_samples, n_features] 
ytest = load() 		# Just a vector of n_samples (i.e. the labels)

# Linear regression model - run for each region to find coefficient weights 
linreg = linear_model.LinearRegression() 

# Train / fit model
linreg.fit(x_test, y_test) 

# Get weights of features in matrix version 
weights = linreg.coef_ 	# Of shape (1, n_features)
print("Feature weights: ", weights) 
print("Temperature weight: ", weights[0])
print("Humidity weight: ", weights[1])
print("Wind speed weight: ", weights[2])
print("Pressure weight: ", weights[3]) 

# Select feature importance (compare to weights found)
modelLin = SelectFromModel(linreg, prefit=True)
X_new_lin = modelLin.transform(x_test)
print("New X shape: ", X_new_lin.shape)


