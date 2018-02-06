import numpy as np
from sklearn import linear_model, datasets
#-------------------------------------------------------------------------
'''
    Big Data Analytics - Project 1
    Resource: http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression.predict

'''

#-------------------------------------------------------------------------
# EXAMPLE DATA AND ANALYSIS ---------------------------------------------#
# # import some data to play with
# iris = datasets.load_iris()
# X = iris.data[:, :2]  # we only take the first two features.
# Y = iris.target

# print("X: ", X)
# print ("Y:", Y)

# h = .02  # step size in the mesh

# logreg = linear_model.LogisticRegression(C=1e5)

# # we create an instance of Neighbours Classifier and fit the data.
# logreg.fit(X, Y)
# print("Coefficient matrix:", logreg.coef_)
#-------------------------------------------------------------------------




# Samples: classified tweets (polarity from -1 to 1)
# Features: weather features such as humidity, pressure, temperature, weather description(rain, clouds, clear etc.)
xtest = load() # matrix of shape[n_samples, n_features] 
ytest = load() # just a vector of n_samples 

# Example - run for each region (might want to change C)
logreg = linear_model.LogisticRegression(C=1e5)

# Train / fit model
logreg.fit(x_test, y_test)
# Get weights of features in matrix version 
weights = logreg.coef_ 	# Of shape (1, n_features)
print("Feature weights: ", weights)

# Find feature importance 
model = SelectFromModel(logreg, prefit=True)

