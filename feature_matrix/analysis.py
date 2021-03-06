import numpy as np
from sklearn import linear_model, datasets
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from scipy import stats
import pickle 

#-------------------------------------------------------------------------
'''
    Big Data Analytics - Project 1
'''


#-------------------------------------------------------------------------

# Samples: classified tweets (polarity from -1 to 1)
# Features: weather features such as humidity, pressure, temperature, wind speed

# Files to train the linear regression model
KansasCity_f = open('KansasCity_feature_matrix.pkl', 'rb')		# Open location specific file 
StLouis_f = open('SaintLouis_feature_matrix.pkl', 'rb')

Atlanta_f = open('Atlanta_feature_matrix.pkl', 'rb')
#Orlando_f = open('Orlando_feature_matrix.pkl', 'rb')

Seattle_f = open('Seattle_feature_matrix.pkl', 'rb')
Portland_f = open('Portland_feature_matrix.pkl', 'rb')

SanDiego_f = open('SanDiego_feature_matrix.pkl', 'rb')
Phoenix_f = open('Phoenix_feature_matrix.pkl', 'rb')

Boston_f = open('Boston_feature_matrix.pkl', 'rb')
#NewYork_f = open('KansasCity_feature_matrix.pkl', 'rb')

# Files to use when predicting sentiment - FILL IN LATER 


try: 
	# Get feature matrix from pickled files to create linear regression model
	KC_file_matrix = pickle.load(KansasCity_f)
	SL_file_matrix = pickle.load(StLouis_f)
	A_file_matrix = pickle.load(Atlanta_f)
	SE_file_matrix = pickle.load(Seattle_f)
	P_file_matrix = pickle.load(Portland_f)
	SD_file_matrix = pickle.load(SanDiego_f)
	PH_file_matrix = pickle.load(Phoenix_f)
	B_file_matrix = pickle.load(Boston_f) 

	# Get feature matrix from pickled files to predict from model
	#KC_file = 
	
	# Convert each file into a matrix
	# Each city gets a matrix called CITY_xtest that has just the weather features
	# Then each city gets a matrix called CITY_ytest that has the samples (i.e. sentiment polarity)
	KC_feature_matrix = np.matrix(KC_file_matrix)
	KC_xtest = KC_feature_matrix[:, 2:6] 		# Matrix of shape[n_samples, n_features] (i.e. the weather)
	KC_X = np.array(KC_xtest, dtype=float)
	KC_ytest = KC_feature_matrix[:,6] 		# Just a vector of n_samples (i.e. the sentiment polarity labels)
	KC_Y = np.array(KC_ytest, dtype=float)
	# PRINT TEST STATEMENTS
	# print(KC_feature_matrix)
	# print("KC_X", KC_X)
	# print("KC_Y", KC_Y)

	SL_feature_matrix = np.matrix(SL_file_matrix)
	SL_xtest = SL_feature_matrix[:, 2:6] 		# Matrix of shape[n_samples, n_features] (i.e. the weather)
	SL_X = np.array(SL_xtest, dtype=float)
	SL_ytest = SL_feature_matrix[:,6] 		# Just a vector of n_samples (i.e. the sentiment polarity labels)
	SL_Y = np.array(SL_ytest, dtype=float) 

	A_feature_matrix = np.matrix(A_file_matrix)
	A_xtest = A_feature_matrix[:, 2:6] 		# Matrix of shape[n_samples, n_features] (i.e. the weather)
	A_X = np.array(A_xtest, dtype=float)
	A_ytest = A_feature_matrix[:,6] 		# Just a vector of n_samples (i.e. the sentiment polarity labels)
	A_Y = np.array(A_ytest, dtype=float) 

	SE_feature_matrix = np.matrix(SE_file_matrix)
	SE_xtest = SE_feature_matrix[:, 2:6] 		# Matrix of shape[n_samples, n_features] (i.e. the weather)
	SE_X = np.array(SE_xtest, dtype=float)
	SE_ytest = SE_feature_matrix[:,6] 		# Just a vector of n_samples (i.e. the sentiment polarity labels)
	SE_Y = np.array(SE_ytest, dtype=float) 

	P_feature_matrix = np.matrix(P_file_matrix)
	P_xtest = P_feature_matrix[:, 2:6] 		# Matrix of shape[n_samples, n_features] (i.e. the weather)
	P_X = np.array(P_xtest, dtype=float)
	P_ytest = P_feature_matrix[:,6] 		# Just a vector of n_samples (i.e. the sentiment polarity labels)
	P_Y = np.array(P_ytest, dtype=float) 

	SD_feature_matrix = np.matrix(SD_file_matrix)
	SD_xtest = SD_feature_matrix[:, 2:6] 		# Matrix of shape[n_samples, n_features] (i.e. the weather)
	SD_X = np.array(SD_xtest, dtype=float)
	SD_ytest = SD_feature_matrix[:,6] 		# Just a vector of n_samples (i.e. the sentiment polarity labels)
	SD_Y = np.array(SD_ytest, dtype=float) 

	PH_feature_matrix = np.matrix(PH_file_matrix)
	PH_xtest = PH_feature_matrix[:, 2:6] 		# Matrix of shape[n_samples, n_features] (i.e. the weather)
	PH_X = np.array(PH_xtest, dtype=float)
	PH_ytest = PH_feature_matrix[:,6] 		# Just a vector of n_samples (i.e. the sentiment polarity labels)
	PH_Y = np.array(PH_ytest, dtype=float) 

	B_feature_matrix = np.matrix(B_file_matrix)
	B_xtest = B_feature_matrix[:, 2:6] 		# Matrix of shape[n_samples, n_features] (i.e. the weather)
	B_X = np.array(B_xtest, dtype=float)
	B_ytest = B_feature_matrix[:,6] 		# Just a vector of n_samples (i.e. the sentiment polarity labels)
	B_Y = np.array(B_ytest, dtype=float) 

	# All sample data that we will be using combined 
	complete_matrix = np.concatenate((KC_X, SL_X, A_X, SE_X, P_X, SD_X, PH_X, B_X))
	zscore_matrix = stats.zscore(complete_matrix)
	print("Zscore matrix shape: ", zscore_matrix.shape)

	# PRINT STATEMENTS FOR TESTING 
	#print(file_matrix[0]) 
	#print("X", X[0])
	#print("ytest", ytest[0])

	# Calculate which city samples are in which rows of the zscore matrix 
	midwest_samples = (len(KC_X) + len(SL_X)) 
		#print("Midwest:", 0, " ", midwest_samples)
	southeast_samples = len(A_X) + midwest_samples
		#print("Southeast range:", midwest_samples, " ", southeast_samples)
	northwest_samples = len(SE_X) + len(P_X) + southeast_samples
		#print("Northwest range:", southeast_samples, " ", northwest_samples)
	southwest_samples = len(SD_X) + len(PH_X) + northwest_samples
		#print("Southwest range:", northwest_samples, " ", southwest_samples)
	northeast_samples = len(B_X) + southwest_samples
		#print("Northeast range:", southwest_samples, " ", northeast_samples)

	# LINEAR REGRESSION MODEL
	# Linear regression model - run for each city to find coefficient weights 
	Midwest_linreg = linear_model.LinearRegression() 
	Southeast_linreg = linear_model.LinearRegression() 
	Northwest_linreg = linear_model.LinearRegression() 
	Southwest_linreg = linear_model.LinearRegression() 
	Northeast_linreg = linear_model.LinearRegression() 

	# Combine samples data into one matrix for each region
	Midwest_Y = np.concatenate((KC_Y, SL_Y))
	Southeast_Y = A_Y
	Northwest_Y = np.concatenate((SE_Y, P_Y))
	Southwest_Y = np.concatenate((SD_Y, PH_Y))
	Northeast_Y = B_Y 

	# Train / fit model 
	# Each region may have one or two cities from which to get samples
	#     Hence why the samples are concatenated below when fitting the data 
	Midwest_linreg.fit(zscore_matrix[0:midwest_samples], Midwest_Y) 
	Southeast_linreg.fit(zscore_matrix[midwest_samples:southeast_samples], Southeast_Y) 
	Northwest_linreg.fit(zscore_matrix[southeast_samples:northwest_samples], Northwest_Y) 
	Southwest_linreg.fit(zscore_matrix[northwest_samples:southwest_samples], Southwest_Y)
	Northeast_linreg.fit(zscore_matrix[southwest_samples:northeast_samples], Northeast_Y) 

	# Get weights of features in linear regression model 
	Midwest_weights = Midwest_linreg.coef_ 	# Of shape (1, n_features) 
	Southeast_weights = Southeast_linreg.coef_ 	# Of shape (1, n_features) 
	Northwest_weights = Northwest_linreg.coef_ 	# Of shape (1, n_features) 
	Southwest_weights = Southwest_linreg.coef_ 	# Of shape (1, n_features) 
	Northeast_weights = Northeast_linreg.coef_ 	# Of shape (1, n_features) 
	
	# RANDOM FOREST CLASSIFIER
	Midwest_clf = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)
	Southeast_clf = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)
	Northwest_clf = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)
	Southwest_clf = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)
	Northeast_clf = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)
	
	a = np.concatenate((KC_Y, SL_Y)).ravel()
	a[a < 0] = -1
	a[a > 0] = 1
	
	b = A_Y.ravel()
	b[b < 0] = -1
	b[b > 0] = 1

	c = np.concatenate((SE_Y, P_Y)).ravel()
	c[c < 0] = -1
	c[c > 0] = 1

	d = np.concatenate((SD_Y, PH_Y)).ravel()
	d[d < 0] = -1
	d[d > 0] = 1

	e = B_Y.ravel()
	e[e < 0] = -1
	e[e > 0] = 1
	Midwest_clf.fit(zscore_matrix[0:midwest_samples], a)
	Southeast_clf.fit(zscore_matrix[midwest_samples:southeast_samples], b) 
	Northwest_clf.fit(zscore_matrix[southeast_samples:northwest_samples], c) 
	Southwest_clf.fit(zscore_matrix[northwest_samples:southwest_samples], d)
	Northeast_clf.fit(zscore_matrix[southwest_samples:northeast_samples], e) 

	# Feature labels to be used so that weather features are ordered 
	feat_labels = ['Humidity','Pressure','Temperature','Wind Speed']
	
	# Select feature importance (compare to weights found)
	MW_sfm = SelectFromModel(Midwest_linreg)
	MW_sfm.fit(zscore_matrix[0:midwest_samples], np.concatenate((KC_Y, SL_Y)))
	print("\nMidwest:")
	print("     Linear reg feature weights: ", Midwest_weights) 
	for feature_list_index in MW_sfm.get_support(indices=True): 
	 	print("    ",  feat_labels[feature_list_index] ) 
	print("     Random forest important features:")
	for feature in zip(feat_labels, Midwest_clf.feature_importances_):
	    print("    ", feature)
	
	SE_sfm = SelectFromModel(Southeast_linreg)
	SE_sfm.fit(zscore_matrix[midwest_samples:southeast_samples], A_Y)
	print("\nSoutheast:")
	print("     Linear reg feature weights: ", Southeast_weights) 
	for feature_list_index in SE_sfm.get_support(indices=True): 
	 	print("    ",  feat_labels[feature_list_index] ) 
	print("     Random forest important features:")
	for feature in zip(feat_labels, Southeast_clf.feature_importances_):
	    print("    ", feature)

	NW_sfm = SelectFromModel(Northwest_linreg)
	NW_sfm.fit(zscore_matrix[southeast_samples:northwest_samples], np.concatenate((SE_Y, P_Y)))
	print("\nNorthwest:")
	print("     Linear reg feature weights: ", Northwest_weights) 
	for feature_list_index in NW_sfm.get_support(indices=True): 
	 	print("    ",  feat_labels[feature_list_index] ) 
	print("     Random forest important features:")
	for feature in zip(feat_labels, Northwest_clf.feature_importances_):
	    print("    ", feature)

	SW_sfm = SelectFromModel(Southwest_linreg)
	SW_sfm.fit(zscore_matrix[northwest_samples:southwest_samples], np.concatenate((SD_Y, PH_Y)))
	print("\nSouthwest:")
	print("     Linear reg feature weights: ", Southwest_weights) 
	for feature_list_index in SW_sfm.get_support(indices=True): 
	 	print("    ",  feat_labels[feature_list_index] ) 
	print("     Random forest important features:")
	for feature in zip(feat_labels, Southwest_clf.feature_importances_):
	    print("    ", feature)

	NE_sfm = SelectFromModel(Northeast_linreg)
	NE_sfm.fit(zscore_matrix[southwest_samples:northeast_samples], B_Y)
	print("\nNortheast:")
	print("     Linear reg feature weights: ", Northeast_weights) 
	for feature_list_index in NE_sfm.get_support(indices=True): 
	 	print("    ", feat_labels[feature_list_index] ) 
	print("     Random forest important features:")
	for feature in zip(feat_labels, Northeast_clf.feature_importances_):
	    print("    ", feature)


	# Find most popular sentiment in city tweets for specified time or hour 
	# Load weather files for each city using example below 
	KC_matrix = np.matrix(KC_file)
	KC_x = KC_matrix[:, 2:6] 		# Matrix of shape[n_samples, n_features] (i.e. the weather)
	KC_polarity = np.array(KC_xx, dtype=float)
	KC_y = KC_feature_matrix[:,6] 		# Just a vector of n_samples (i.e. the sentiment polarity labels)
	KC_weather = np.array(KC_y, dtype=float)

	SL_weather = 0

	# Concatenate matrices by region with just the weather data 
	Midwest_weather = np.concatenate((KC_weather, SL_weather))
	Southeast_weather = np.concatenate((KC_weather, SL_weather))
	Northwest_weather = np.concatenate((KC_weather, SL_weather))
	Southwest_weather = np.concatenate((KC_weather, SL_weather))
	Northeast_weather = np.concatenate((KC_weather, SL_weather))
	
	# Run linear regression model on each region
	Midwest_sentiment = Midwest_linreg.predict(Midwest_weather)
	Southeast_sentiment = Southeast_linreg.predict(Southeast_weather)
	Northwest_sentiment = Northwest_linreg.predict(Northwest_weather)
	Southwest_sentiment = Southwest_linreg.predict(Southwest_weather)
	Northeast_sentiment = Northeast_linreg.predict(Northeast_weather)

	# Convert polarity to discrete numbers
	test[test < 0] = -1
	test[test > 0] = 1
	num_neutral = (test == 0).sum()	# Find the sum of neg/pos/neutral sentiments 
	num_pos = (test == 1).sum()
	num_neg = (test == -1).sum()
	sentiments = np.array([num_neg, num_neutral, num_pos])
	maxval = 0
	pop_sentiment = -1
	pop_two_sentiment = -1
	for i in range(len(sentiments)): 
		if sentiments[i] > maxval: 
			pop_sentiment = i
			maxval = sentiments[i]
		elif sentiments[i] == maxval: 
			pop_two_sentiment = i
	
	popular_Midwest = "" 
	if (pop_sentiment == 0): 
		popular = "Negative"
	elif (pop_sentiment == 1): 
		popular = "Neutral"
	else: 
		popular = "Positive"

	# Print most popular sentiment
	print("Most popular sentiment test:", popular)





finally:
	KansasCity_f.close()		# Close files
	StLouis_f.close()

	Atlanta_f.close()
	#Orlando_f.close()

	Seattle_f.close()
	Portland_f.close()

	SanDiego_f.close()
	Phoenix_f.close()

	Boston_f.close()
	#NewYork_f.close()




#-------------------------------------------------------------------------
# EXAMPLE DATA AND ANALYSIS 

# Find most popular sentiment in city tweets for specified time or hour 
	# test = np.array([0., 0.2, 0.2, 0.05, 0.05, 0.5, -0.5, -0.5, 0.5, -0.8, -0.8, -0.9])
	# test2 = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 
	# 	0.2, 0.2, 0.05, 0.05, 0.5, -0.5, -0.5, 0.5, -0.8, -0.8, -0.9])
	# test.astype(float)
	# # Convert polarity to discrete numbers
	# test[test < 0] = -1
	# test[test > 0] = 1
	# num_neutral = (test == 0).sum()	# Find the sum of neg/pos/neutral sentiments 
	# num_pos = (test == 1).sum()
	# num_neg = (test == -1).sum()
	# # print("Negative:", num_neg, " Neutral:", num_neutral, " Positive:", num_pos)
	# sentiments = np.array([num_neg, num_neutral, num_pos])
	# maxval = 0
	# pop_sentiment = -1
	# pop_two_sentiment = -1
	# for i in range(len(sentiments)): 
	# 	if sentiments[i] > maxval: 
	# 		pop_sentiment = i
	# 		maxval = sentiments[i]
	# 	elif sentiments[i] == maxval: 
	# 		pop_two_sentiment = i

	# popular_Midwest = "" 
	# if (pop_sentiment == 0): 
	# 	popular = "Negative"
	# elif (pop_sentiment == 1): 
	# 	popular = "Neutral"
	# else: 
	# 	popular = "Positive"

	# # Print most popular sentiment
	# print("Most popular sentiment test:", popular)

# OLD DATA TYPE
# dt = np.dtype([('city', object, 11), ('datetime', np.unicode_, 19), 
# 		('w1', np.float32, 6),
# 		('w2', np.float32, 6),
# 		('w3', np.float32, 6),
# 		('w4', np.float32, 6),
# 		('sentiment', np.float32, 10)])

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




