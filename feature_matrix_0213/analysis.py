import numpy as np
from sklearn import linear_model, datasets
from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
from scipy import stats
from math import sqrt
import pandas as pd 
import pickle 

#-------------------------------------------------------------------------
'''
    Big Data Analytics - Project 1
'''


#-------------------------------------------------------------------------

def findPopularSentiment(sent_array): 
	num_neutral = (sent_array == 0).sum()	# Find the sum of neg/pos/neutral sentiments 
	num_pos = (sent_array == 1).sum()
	num_neg = (sent_array == -1).sum()
	sentiments = np.array([num_neg, num_neutral, num_pos])
	maxval = np.argmax(sentiments)
	if (maxval == 0): 
		return "Negative"
	elif (maxval == 2): 
		return "Positive"
	else: 
		return "Neutral"


# Samples: classified tweets (polarity from -1 to 1)
# Features: weather features such as humidity, pressure, temperature, wind speed

# Files to train the linear regression model
KansasCity_f = open('KansasCity_feature_matrix.pkl', 'rb')		# Open location specific file 
StLouis_f = open('SaintLouis_feature_matrix.pkl', 'rb')
Atlanta_f = open('Atlanta_feature_matrix.pkl', 'rb')
Seattle_f = open('Seattle_feature_matrix.pkl', 'rb')
Portland_f = open('Portland_feature_matrix.pkl', 'rb')
SanDiego_f = open('SanDiego_feature_matrix.pkl', 'rb')
Phoenix_f = open('Phoenix_feature_matrix.pkl', 'rb')
Boston_f = open('Boston_feature_matrix.pkl', 'rb')
NewYork_f = open('NewYork_feature_matrix.pkl', 'rb')

# # Files with full matrix 
# KansasCity_full = open('KansasCity_feature_matrix_full.pkl', 'rb')		# Open location specific file 
# StLouis_full = open('SaintLouis_feature_matrix_full.pkl', 'rb')
# Atlanta_full = open('Atlanta_feature_matrix_full.pkl', 'rb')
# Seattle_full = open('Seattle_feature_matrix_full.pkl', 'rb')
# Portland_full = open('Portland_feature_matrix_full.pkl', 'rb')
# SanDiego_full = open('SanDiego_feature_matrix_full.pkl', 'rb')
# Phoenix_full = open('Phoenix_feature_matrix_full.pkl', 'rb')
# Boston_full = open('Boston_feature_matrix_full.pkl', 'rb')
# NewYork_full = open('NewYork_feature_matrix_full.pkl', 'rb')

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
	NY_file_matrix = pickle.load(NewYork_f) 
	
	# Convert each file into a matrix
	# Each city gets a matrix called CITY_xtest that has just the weather features
	# Then each city gets a matrix called CITY_ytest that has the samples (i.e. sentiment polarity)
	
	#------------- MIDWEST SAMPLES -------------#
	KC_feature_matrix = np.matrix(KC_file_matrix)
	KC_xinit = KC_feature_matrix[:, 2:6] 		# Matrix of shape[n_samples, n_features] (i.e. the weather)
	KC_X = np.array(KC_xinit, dtype=float)
	KC_yinit = KC_feature_matrix[:,6] 		# Just a vector of n_samples (i.e. the sentiment polarity labels)
	KC_Y = np.array(KC_yinit, dtype=float)
	# PRINT TEST STATEMENTS
	# print(KC_feature_matrix)
	# print("KC_X", KC_X)
	# print("KC_Y", KC_Y)

	SL_feature_matrix = np.matrix(SL_file_matrix)
	SL_xinit = SL_feature_matrix[:, 2:6] 		# Matrix of shape[n_samples, n_features] (i.e. the weather)
	SL_X = np.array(SL_xinit, dtype=float)
	SL_yinit = SL_feature_matrix[:,6] 		# Just a vector of n_samples (i.e. the sentiment polarity labels)
	SL_Y = np.array(SL_yinit, dtype=float) 

	Midwest_zmatrix = stats.zscore(np.concatenate((KC_X, SL_X)))
	Midwest_Y = np.concatenate((KC_Y, SL_Y))
	MW_X_train, MW_X_test, MW_y_train, MW_y_test = train_test_split(Midwest_zmatrix, Midwest_Y, test_size=0.20, random_state=42)


	#------------- SOUTHEAST SAMPLES -------------#
	A_feature_matrix = np.matrix(A_file_matrix)
	A_xinit = A_feature_matrix[:, 2:6] 		# Matrix of shape[n_samples, n_features] (i.e. the weather)
	A_X = np.array(A_xinit, dtype=float)
	A_yinit = A_feature_matrix[:,6] 		# Just a vector of n_samples (i.e. the sentiment polarity labels)
	A_Y = np.array(A_yinit, dtype=float) 
	
	Southeast_zmatrix = stats.zscore(A_X)
	SE_X_train, SE_X_test, SE_y_train, SE_y_test = train_test_split(Southeast_zmatrix, A_Y, test_size=0.20, random_state=42)


	#------------- NORTHWEST SAMPLES -------------#
	SE_feature_matrix = np.matrix(SE_file_matrix)
	SE_xinit = SE_feature_matrix[:, 2:6] 		# Matrix of shape[n_samples, n_features] (i.e. the weather)
	SE_X = np.array(SE_xinit, dtype=float)
	SE_yinit = SE_feature_matrix[:,6] 		# Just a vector of n_samples (i.e. the sentiment polarity labels)
	SE_Y = np.array(SE_yinit, dtype=float) 

	P_feature_matrix = np.matrix(P_file_matrix)
	P_xinit = P_feature_matrix[:, 2:6] 		# Matrix of shape[n_samples, n_features] (i.e. the weather)
	P_X = np.array(P_xinit, dtype=float)
	P_yinit = P_feature_matrix[:,6] 		# Just a vector of n_samples (i.e. the sentiment polarity labels)
	P_Y = np.array(P_yinit, dtype=float) 

	Northwest_zmatrix = stats.zscore(np.concatenate((SE_X, P_X)))
	Northwest_Y = np.concatenate((SE_Y, P_Y))
	NW_X_train, NW_X_test, NW_y_train, NW_y_test = train_test_split(Northwest_zmatrix, Northwest_Y, test_size=0.20, random_state=42)


	#------------- SOUTHWEST SAMPLES -------------#
	SD_feature_matrix = np.matrix(SD_file_matrix)
	SD_xinit = SD_feature_matrix[:, 2:6] 		# Matrix of shape[n_samples, n_features] (i.e. the weather)
	SD_X = np.array(SD_xinit, dtype=float)
	SD_yinit = SD_feature_matrix[:,6] 		# Just a vector of n_samples (i.e. the sentiment polarity labels)
	SD_Y = np.array(SD_yinit, dtype=float) 

	PH_feature_matrix = np.matrix(PH_file_matrix)
	PH_xinit = PH_feature_matrix[:, 2:6] 		# Matrix of shape[n_samples, n_features] (i.e. the weather)
	PH_X = np.array(PH_xinit, dtype=float)
	PH_yinit = PH_feature_matrix[:,6] 		# Just a vector of n_samples (i.e. the sentiment polarity labels)
	PH_Y = np.array(PH_yinit, dtype=float) 

	Southwest_zmatrix = stats.zscore(np.concatenate((SD_X, PH_X)))
	Southwest_Y = np.concatenate((SD_Y, PH_Y))
	SW_X_train, SW_X_test, SW_y_train, SW_y_test = train_test_split(Southwest_zmatrix, Southwest_Y, test_size=0.20, random_state=42)
	

	#------------- NORTHEAST SAMPLES -------------#
	B_feature_matrix = np.matrix(B_file_matrix)
	B_xinit = B_feature_matrix[:, 2:6] 		# Matrix of shape[n_samples, n_features] (i.e. the weather)
	B_X = np.array(B_xinit, dtype=float)
	B_yinit = B_feature_matrix[:,6] 		# Just a vector of n_samples (i.e. the sentiment polarity labels)
	B_Y = np.array(B_yinit, dtype=float) 

	NY_feature_matrix = np.matrix(NY_file_matrix)
	NY_xinit = NY_feature_matrix[:, 2:6] 		# Matrix of shape[n_samples, n_features] (i.e. the weather)
	NY_X = np.array(NY_xinit, dtype=float)
	NY_yinit = NY_feature_matrix[:,6] 		# Just a vector of n_samples (i.e. the sentiment polarity labels)
	NY_Y = np.array(NY_yinit, dtype=float) 

	Northeast_zmatrix = stats.zscore(np.concatenate((B_X, NY_X)))
	Northeast_Y = np.concatenate((B_Y, NY_Y))
	NE_X_train, NE_X_test, NE_y_train, NE_y_test = train_test_split(Northeast_zmatrix, Northeast_Y, test_size=0.20, random_state=42)



	#------------- LINEAR REGRESSION MODEL -------------#
	# Linear regression model - run for each city to find coefficient weights 
	Midwest_linreg = linear_model.LinearRegression() 
	Southeast_linreg = linear_model.LinearRegression() 
	Northwest_linreg = linear_model.LinearRegression() 
	Southwest_linreg = linear_model.LinearRegression() 
	Northeast_linreg = linear_model.LinearRegression() 

	# FIT TRAINING DATA
	Midwest_linreg.fit(MW_X_train, MW_y_train) 
	Southeast_linreg.fit(SE_X_train, SE_y_train) 
	Northwest_linreg.fit(NW_X_train, NW_y_train) 
	Southwest_linreg.fit(SW_X_train, SW_y_train)
	Northeast_linreg.fit(NE_X_train, NE_y_train) 

	# Get weights of features in linear regression model 
	Midwest_weights = Midwest_linreg.coef_ 	# Of shape (1, n_features) 
	Southeast_weights = Southeast_linreg.coef_ 	# Of shape (1, n_features) 
	Northwest_weights = Northwest_linreg.coef_ 	# Of shape (1, n_features) 
	Southwest_weights = Southwest_linreg.coef_ 	# Of shape (1, n_features) 
	Northeast_weights = Northeast_linreg.coef_ 	# Of shape (1, n_features) 

	# Predict now 
	MW_pred = Midwest_linreg.predict(MW_X_test)
	SE_pred = Southeast_linreg.predict(SE_X_test)
	NW_pred = Northwest_linreg.predict(NW_X_test)
	SW_pred = Southwest_linreg.predict(SW_X_test)
	NE_pred = Northeast_linreg.predict(NE_X_test)
	#print("Midwest prediction:", MW_pred)

	# If we want to calculate RMSE
	MW_rmse = sqrt(mean_squared_error(MW_y_test, MW_pred))
	SE_rmse = sqrt(mean_squared_error(SE_y_test, SE_pred))
	NW_rmse = sqrt(mean_squared_error(NW_y_test, NW_pred))
	SW_rmse = sqrt(mean_squared_error(SW_y_test, SW_pred))
	NE_rmse = sqrt(mean_squared_error(NE_y_test, NE_pred))
	print("Midwest rmse", MW_rmse) 
	print("Southeast rmse", SE_rmse) 
	print("Northwest rmse", NW_rmse) 
	print("Southwest rmse", SW_rmse) 
	print("Northeast rmse", NE_rmse) 


	# Save each true value matrix to csv
	MW_df_act = pd.DataFrame(MW_y_test)
	MW_df_act.to_csv("MW_y_true.csv")

	SE_df_act = pd.DataFrame(MW_y_test)
	SE_df_act.to_csv("SE_y_true.csv")

	NW_df_act = pd.DataFrame(NW_y_test)
	NW_df_act.to_csv("NW_y_true.csv")

	SW_df_act = pd.DataFrame(SW_y_test)
	SW_df_act.to_csv("SW_y_true.csv")
	
	NE_df_act = pd.DataFrame(NE_y_test)
	NE_df_act.to_csv("NE_y_true.csv")



	# Save each prediction matrix to csv 
	MW_df_pred = pd.DataFrame(MW_pred)
	MW_df_pred.to_csv("MW_df_pred.csv")

	SE_df_pred = pd.DataFrame(SE_pred)
	SE_df_pred.to_csv("SE_df_pred.csv")

	NW_df_pred = pd.DataFrame(NW_pred)
	NW_df_pred.to_csv("NW_df_pred.csv")

	SW_df_pred = pd.DataFrame(SW_pred)
	SW_df_pred.to_csv("SW_df_pred.csv")

	NE_df_pred = pd.DataFrame(NE_pred)
	NE_df_pred.to_csv("NE_df_pred.csv")


	
	#------------- RANDOM FOREST CLASSIFIER -------------#

	# # Load full feature matrix from pickled files to create random forest classifier
	# KC_full_matrix = pickle.load(KansasCity_full)
	# SL_full_matrix = pickle.load(StLouis_full)
	# A_full_matrix = pickle.load(Atlanta_full)
	# SE_full_matrix = pickle.load(Seattle_full)
	# P_full_matrix = pickle.load(Portland_full)
	# SD_full_matrix = pickle.load(SanDiego_full)
	# PH_full_matrix = pickle.load(Phoenix_full)
	# B_full_matrix = pickle.load(Boston_full) 
	# NY_full_matrix = pickle.load(NewYork_full)


	# #------------- MIDWEST SAMPLES -------------#
	# KC_full_fmatrix = np.matrix(KC_full_matrix)
	# KC_fin_matrix = np.delete(KC_full_fmatrix, 3, 1)
	# KC_xinit_full = KC_fin_matrix[:, 2:14] 		# Matrix of shape[n_samples, n_features] (i.e. the weather)
	# KC_X_full = np.array(KC_xinit_full, dtype=float)
	# KC_yinit_full = KC_fin_matrix[:,14] 		# Just a vector of n_samples (i.e. the sentiment polarity labels)
	# KC_Y_full = np.array(KC_yinit_full, dtype=float) 

	# SL_full_fmatrix = np.matrix(SL_full_matrix)
	# SL_fin_matrix = np.delete(SL_full_fmatrix, 3, 1)
	# SL_xinit_full = SL_fin_matrix[:, 2:14] 		# Matrix of shape[n_samples, n_features] (i.e. the weather)
	# SL_X_full = np.array(SL_xinit_full, dtype=float)
	# SL_yinit_full = SL_fin_matrix[:,14] 		# Just a vector of n_samples (i.e. the sentiment polarity labels)
	# SL_Y_full = np.array(SL_yinit_full, dtype=float) 

	# #Midwest_clf_x = stats.zscore(np.concatenate((KC_X_full, SL_X_full)))
	# Midwest_clf_x = np.concatenate((KC_X_full, SL_X_full))
	# Midwest_clf_y = np.concatenate((KC_Y_full, SL_Y_full)).ravel()
	# Midwest_clf_y[Midwest_clf_y < 0] = -1
	# Midwest_clf_y[Midwest_clf_y > 0] = 1

	# MW_X_train, MW_X_test, MW_y_train, MW_y_test = train_test_split(Midwest_clf_x, Midwest_clf_y, test_size=0.20, random_state=42)



	# #------------- SOUTHEAST SAMPLES -------------#
	# A_full_fmatrix = np.matrix(A_full_matrix)
	# A_fin_matrix = np.delete(A_full_fmatrix, 3, 1)
	# A_xinit_full = A_fin_matrix[:, 2:14] 		# Matrix of shape[n_samples, n_features] (i.e. the weather)
	# A_X_full = np.array(A_xinit_full, dtype=float)
	# A_yinit_full = A_fin_matrix[:,14] 		# Just a vector of n_samples (i.e. the sentiment polarity labels)
	# A_Y_full = np.array(A_yinit_full, dtype=float) 

	# # print("A_x full shape", A_X_full.shape)
	# # print("A_X_full", A_X_full)
	# # print("A_y full shape", A_Y_full.shape)
	# # print("A_Y_full.ravel()", A_Y_full.ravel())

	# #Southeast_clf_x = stats.zscore(A_X_full)
	# Southeast_clf_x = A_X_full
	# Southeast_clf_y = A_Y_full.ravel()
	# Southeast_clf_y[Southeast_clf_y < 0] = -1
	# Southeast_clf_y[Southeast_clf_y > 0] = 1

	# SE_X_train, SE_X_test, SE_y_train, SE_y_test = train_test_split(Southeast_clf_x, Southeast_clf_y, test_size=0.20, random_state=42)



	# #------------- NORTHWEST SAMPLES -------------#
	# SE_full_fmatrix = np.matrix(SE_full_matrix)
	# SE_fin_matrix = np.delete(SE_full_fmatrix, 3, 1)
	# SE_xinit_full = SE_fin_matrix[:, 2:14] 		# Matrix of shape[n_samples, n_features] (i.e. the weather)
	# SE_X_full = np.array(SE_xinit_full, dtype=float)
	# SE_yinit_full = SE_fin_matrix[:,14] 		# Just a vector of n_samples (i.e. the sentiment polarity labels)
	# SE_Y_full = np.array(SE_yinit_full, dtype=float) 

	# P_full_fmatrix = np.matrix(P_full_matrix)
	# P_fin_matrix = np.delete(P_full_fmatrix, 3, 1)
	# P_xinit_full = P_fin_matrix[:, 2:14] 		# Matrix of shape[n_samples, n_features] (i.e. the weather)
	# P_X_full = np.array(P_xinit_full, dtype=float)
	# P_yinit_full = P_fin_matrix[:,14] 		# Just a vector of n_samples (i.e. the sentiment polarity labels)
	# P_Y_full = np.array(P_yinit_full, dtype=float) 


	# #Northwest_clf_x = stats.zscore(np.concatenate((SE_X_full, P_X_full)))
	# Northwest_clf_x = np.concatenate((SE_X_full, P_X_full))
	# Northwest_clf_y = np.concatenate((SE_Y_full, P_Y_full)).ravel()
	# Northwest_clf_y[Northwest_clf_y < 0] = -1
	# Northwest_clf_y[Northwest_clf_y > 0] = 1

	# NW_X_train, NW_X_test, NW_y_train, NW_y_test = train_test_split(Northwest_clf_x, Northwest_clf_y, test_size=0.20, random_state=42)



	# #------------- SOUTHWEST SAMPLES -------------#
	# SD_full_fmatrix = np.matrix(SD_full_matrix)
	# SD_fin_matrix = np.delete(SD_full_fmatrix, 3, 1)
	# SD_xinit_full = SD_fin_matrix[:, 2:14] 		# Matrix of shape[n_samples, n_features] (i.e. the weather)
	# SD_X_full = np.array(SD_xinit_full, dtype=float)
	# SD_yinit_full = SD_fin_matrix[:,14] 		# Just a vector of n_samples (i.e. the sentiment polarity labels)
	# SD_Y_full = np.array(SD_yinit_full, dtype=float) 

	# PH_full_fmatrix = np.matrix(PH_full_matrix)
	# PH_fin_matrix = np.delete(PH_full_fmatrix, 3, 1)
	# PH_xinit_full = PH_fin_matrix[:, 2:14] 		# Matrix of shape[n_samples, n_features] (i.e. the weather)
	# PH_X_full = np.array(PH_xinit_full, dtype=float)
	# PH_yinit_full = PH_fin_matrix[:,14] 		# Just a vector of n_samples (i.e. the sentiment polarity labels)
	# PH_Y_full = np.array(PH_yinit_full, dtype=float) 

	# #Southwest_clf_x = stats.zscore(np.concatenate((SD_X_full, PH_X_full)))
	# Southwest_clf_x = np.concatenate((SD_X_full, PH_X_full))
	# Southwest_clf_y = np.concatenate((SD_Y_full, PH_Y_full)).ravel()
	# Southwest_clf_y[Southwest_clf_y < 0] = -1
	# Southwest_clf_y[Southwest_clf_y > 0] = 1

	# SW_X_train, SW_X_test, SW_y_train, SW_y_test = train_test_split(Southwest_clf_x, Southwest_clf_y, test_size=0.20, random_state=42)



	# #------------- NORTHEAST SAMPLES -------------#
	# B_full_fmatrix = np.matrix(B_full_matrix)
	# B_fin_matrix = np.delete(B_full_fmatrix, 3, 1)
	# B_xinit_full = B_fin_matrix[:, 2:14] 		# Matrix of shape[n_samples, n_features] (i.e. the weather)
	# B_X_full = np.array(B_xinit_full, dtype=float)
	# B_yinit_full = B_fin_matrix[:,14] 		# Just a vector of n_samples (i.e. the sentiment polarity labels)
	# B_Y_full = np.array(B_yinit_full, dtype=float) 


	# NY_full_fmatrix = np.matrix(NY_full_matrix)
	# print("NY shape", NY_full_fmatrix.shape)
	# print("NY row", NY_full_fmatrix[0])
	# NY_fin_matrix = np.delete(NY_full_fmatrix, 3, 1)
	# NY_xinit_full = NY_fin_matrix[:, 2:14] 		# Matrix of shape[n_samples, n_features] (i.e. the weather)
	# NY_X_full = np.array(NY_xinit_full, dtype=float)
	# NY_yinit_full = NY_fin_matrix[:,14] 		# Just a vector of n_samples (i.e. the sentiment polarity labels)
	# NY_Y_full = np.array(NY_yinit_full, dtype=float) 

	# # print("B full shape",B_full_fmatrix.shape)
	# # print("B full matrix", B_full_fmatrix[0])
	# # print("B full 2 matrix", B_fin_matrix[0])
	# # print("B x shape",B_X_full[0])
	# # print("B y shape",B_Y_full)
	# Northeast_clf_x = np.concatenate((B_X_full, NY_X_full))
	# Northeast_clf_y = np.concatenate((B_Y_full, NY_Y_full)).ravel()
	# Northeast_clf_y[Northeast_clf_y < 0] = -1
	# Northeast_clf_y[Northeast_clf_y > 0] = 1

	# NE_X_train, NE_X_test, NE_y_train, NE_y_test = train_test_split(Northeast_clf_x, Northeast_clf_y, test_size=0.20, random_state=42)


	# Midwest_clf = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)
	# Southeast_clf = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)
	# Northwest_clf = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)
	# Southwest_clf = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)
	# Northeast_clf = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)
	
	# print("About to run random forest classifier")
	
	# Midwest_clf.fit(MW_X_train, MW_y_train)
	# Southeast_clf.fit(SE_X_train, SE_y_train)
	# Northwest_clf.fit(NW_X_train, NW_y_train) 
	# Southwest_clf.fit(SW_X_train, SW_y_train)
	# Northeast_clf.fit(NE_X_train, NE_y_train)

	#------------- ACTUAL MAJORITY VALUE -------------# 
	# MW_pop_sent = findPopularSentiment(MW_y_test)
	# SE_pop_sent = findPopularSentiment(SE_y_test)
	# NW_pop_sent = findPopularSentiment(NW_y_test)
	# SW_pop_sent = findPopularSentiment(SW_y_test)
	# NE_pop_sent = findPopularSentiment(NE_y_test)	

	#------------- PREDICT USING CLASSIFIER -------------# 
	# MW_pred_sent = Midwest_clf.predict(MW_X_test)
	# SE_pred_sent = Southeast_clf.predict(SE_X_test)
	# NW_pred_sent = Northwest_clf.predict(NW_X_test) 
	# SW_pred_sent = Southwest_clf.predict(SW_X_test)
	# NE_pred_sent = Northeast_clf.predict(NE_X_test)


	# Save each true value matrix to csv
	# NE_df_act = pd.DataFrame(NE_y_test)
	# NE_df_act.to_csv("NE_y_true_clf.csv")

	# Save each prediction matrix to csv 
	# NE_df_pred = pd.DataFrame(NE_pred_sent)
	# NE_df_pred.to_csv("NE_df_pred_clf.csv")


	# #------------- PRINT STATEMENTS FOR LINEAR REGRESSION -------------# 

	# Feature labels to be used so that weather features are ordered 
	feat_labels = ['Humidity','Pressure','Temperature','Wind Speed']

	# Select feature importance (compare to weights found)
	MW_sfm = SelectFromModel(Midwest_linreg)
	MW_sfm.fit(Midwest_zmatrix, Midwest_Y)
	print("\nMidwest:")
	print("     Linear reg feature weights: ", Midwest_weights) 
	for feature_list_index in MW_sfm.get_support(indices=True): 
	 	print("    ",  feat_labels[feature_list_index] ) 
	
	
	SE_sfm = SelectFromModel(Southeast_linreg)
	SE_sfm.fit(Southeast_zmatrix, A_Y)
	print("\nSoutheast:")
	print("     Linear reg feature weights: ", Southeast_weights) 
	for feature_list_index in SE_sfm.get_support(indices=True): 
	 	print("    ",  feat_labels[feature_list_index] ) 
	

	NW_sfm = SelectFromModel(Northwest_linreg)
	NW_sfm.fit(Northwest_zmatrix, Northwest_Y)
	print("\nNorthwest:")
	print("     Linear reg feature weights: ", Northwest_weights) 
	for feature_list_index in NW_sfm.get_support(indices=True): 
	 	print("    ",  feat_labels[feature_list_index] ) 
	

	SW_sfm = SelectFromModel(Southwest_linreg)
	SW_sfm.fit(Southwest_zmatrix, Southwest_Y)
	print("\nSouthwest:")
	print("     Linear reg feature weights: ", Southwest_weights) 
	for feature_list_index in SW_sfm.get_support(indices=True): 
	 	print("    ",  feat_labels[feature_list_index] ) 
	

	NE_sfm = SelectFromModel(Northeast_linreg)
	NE_sfm.fit(Northeast_zmatrix, Northeast_Y)
	print("\nNortheast:")
	print("     Linear reg feature weights: ", Northeast_weights) 
	for feature_list_index in NE_sfm.get_support(indices=True): 
	 	print("    ", feat_labels[feature_list_index] ) 
	


	#------------- PRINT STATEMENTS FOR RANDOM FOREST FEATURE IMPORTANCE -------------# 

	feat_labels_full = ['Humidity','Pressure','Temperature','Wind Speed', 'Clear','Rain','Snow','Sleet','Wind','Fog','Cloudy','Partly Cloudy']

	# print("\nMidwest:")
	# print("Predicted classification: ", MW_pred_sent)
	# print("Actual classification: ", MW_pop_sent)
	# print("     Random forest important features:")
	# for feature in zip(feat_labels_full, Midwest_clf.feature_importances_):
	#     print("    ", feature)


	# print("\nSoutheast:")
	# print("Predicted classification: ", SE_pred_sent)
	# print("Actual classification: ", SE_pop_sent)
	# print("     Random forest important features:")
	# for feature in zip(feat_labels_full, Southeast_clf.feature_importances_):
	#     print("    ", feature)

	# print("\nNorthwest:")
	# print("Predicted classification: ",  NW_pred_sent)
	# print("Actual classification: ", NW_pop_sent)
	# print("     Random forest important features:")
	# for feature in zip(feat_labels_full, Northwest_clf.feature_importances_):
	#     print("    ", feature)


	# print("\nSouthwest:")
	# print("Predicted classification: ", SW_pred_sent)
	# print("Actual classification: ", SW_pop_sent)
	# print("     Random forest important features:")
	# for feature in zip(feat_labels_full, Southwest_clf.feature_importances_):
	#     print("    ", feature)


	# print("\nNortheast:")
	# print("Predicted classification: ", NE_pred_sent)
	# print("Actual classification: ", NE_pop_sent)
	# print("     Random forest important features:")
	# for feature in zip(feat_labels_full, Northeast_clf.feature_importances_):
	#     print("    ", feature)




finally:
	KansasCity_f.close()		# Close files
	StLouis_f.close()
	Atlanta_f.close()
	Seattle_f.close()
	Portland_f.close()
	SanDiego_f.close()
	Phoenix_f.close()
	Boston_f.close()
	NewYork_f.close()

	# KansasCity_full.close()		# Close files
	# StLouis_full.close()
	# Atlanta_full.close()
	# Seattle_full.close()
	# Portland_full.close()
	# SanDiego_full.close()
	# Phoenix_full.close()
	# Boston_full.close()
	# NewYork_full.close()




#-------------------------------------------------------------------------
# EXAMPLE DATA AND ANALYSIS 


# OLD FIND MOST POPULAR SENTIMENT
# Find most popular sentiment in city tweets for specified time or hour 
	# Load weather files for each city using example below 
	# KC_matrix = np.matrix(KC_file)
	# KC_x = KC_matrix[:, 2:6] 		# Matrix of shape[n_samples, n_features] (i.e. the weather)
	# KC_polarity = np.array(KC_xx, dtype=float)
	# KC_y = KC_feature_matrix[:,6] 		# Just a vector of n_samples (i.e. the sentiment polarity labels)
	# KC_weather = np.array(KC_y, dtype=float)

	# SL_weather = 0

	# Concatenate matrices by region with just the weather data 
	# Midwest_weather = np.concatenate((KC_weather, SL_weather))
	# Southeast_weather = np.concatenate((KC_weather, SL_weather))
	# Northwest_weather = np.concatenate((KC_weather, SL_weather))
	# Southwest_weather = np.concatenate((KC_weather, SL_weather))
	# Northeast_weather = np.concatenate((KC_weather, SL_weather))
	
	# Run linear regression model on each region
	# Midwest_sentiment = Midwest_linreg.predict(Midwest_weather)
	# Southeast_sentiment = Southeast_linreg.predict(Southeast_weather)
	# Northwest_sentiment = Northwest_linreg.predict(Northwest_weather)
	# Southwest_sentiment = Southwest_linreg.predict(Southwest_weather)
	# Northeast_sentiment = Northeast_linreg.predict(Northeast_weather)

	# Convert polarity to discrete numbers
	# test[test < 0] = -1
	# test[test > 0] = 1
	# num_neutral = (test == 0).sum()	# Find the sum of neg/pos/neutral sentiments 
	# num_pos = (test == 1).sum()
	# num_neg = (test == -1).sum()
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

	# Print most popular sentiment
	# print("Most popular sentiment test:", popular)




# OLD VERSION OF CALCULATING Z SCORE MATRIX
# All sample data that we will be using combined 
# complete_matrix = np.concatenate((KC_X, SL_X, A_X, SE_X, P_X, SD_X, PH_X, B_X))

# # Calculate which city samples are in which rows of the zscore matrix 
# midwest_samples = (len(KC_X) + len(SL_X)) 
# 	#print("Midwest:", 0, " ", midwest_samples)
# southeast_samples = len(A_X) + midwest_samples
# 	#print("Southeast range:", midwest_samples, " ", southeast_samples)
# northwest_samples = len(SE_X) + len(P_X) + southeast_samples
# 	#print("Northwest range:", southeast_samples, " ", northwest_samples)
# southwest_samples = len(SD_X) + len(PH_X) + northwest_samples
# 	#print("Southwest range:", northwest_samples, " ", southwest_samples)
# northeast_samples = len(B_X) + southwest_samples
# 	#print("Northeast range:", southwest_samples, " ", northeast_samples)

# # Train / fit model - OLD VERSION
	# # Each region may have one or two cities from which to get samples
	# #     Hence why the samples are concatenated below when fitting the data 
	# # Midwest_linreg.fit(zscore_matrix[0:midwest_samples], Midwest_Y) 
	# # Southeast_linreg.fit(zscore_matrix[midwest_samples:southeast_samples], Southeast_Y) 
	# # Northwest_linreg.fit(zscore_matrix[southeast_samples:northwest_samples], Northwest_Y) 
	# # Southwest_linreg.fit(zscore_matrix[northwest_samples:southwest_samples], Southwest_Y)
	# # Northeast_linreg.fit(zscore_matrix[southwest_samples:northeast_samples], Northeast_Y) 







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




