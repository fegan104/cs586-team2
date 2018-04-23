## PCA 

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pandas as pd 
import numpy as np
import plotly.plotly as py
from plotly.graph_objs import *
import plotly.tools as tls
import seaborn as sns

tls.set_credentials_file(username='hanguyen', api_key='0MwWvySm0Y0NpyreLjle')

# Get feature matrix from pickled files to create linear regression model
df = pd.read_excel('cat.xlsx')
df.as_matrix()
X = df.values[:, 0:25] 			# Matrix of shape[n_samples, n_features] (32 features)
y = df.values[:,26] 			# A vector of n_samples (i.e. the price labels)
target_names = ['LAND SQUARE FEET','GROSS SQUARE FEET','YEAR BUILT','commercial','residencial','mixed','# of bus_stop','# of subway_station','# of FELONY','# of VIOLATION','# of MISDEMEANOR','total # of poi','poi type 1','poi type 2','poi type 3','poi type 4','poi type 5','poi type 6','poi type 7','poi type 8','poi type 9','poi type 10','poi type 11','poi type 12','poi type 13','# of roads','total width of raods']


# SCALE DATA 
z_scaler = StandardScaler()
z_data = z_scaler.fit_transform(np.copy(X))


# PCA 
# To getter a better understanding of interaction of the dimensions
# plot the first three PCA dimensions
# pca = PCA(n_components=0.95, svd_solver='full')
# pca.fit(z_data)
# X_reduced = pca.transform(z_data)
# #print("pca explained variance", pca.explained_variance_)
# print("pca explained variance ratio", pca.explained_variance_ratio_)
# #print("pca components", pca.components_)
# print("shape", X_reduced.shape)


# LINEAR DISCRIMINANT ANALYSIS
# lda = LinearDiscriminantAnalysis(n_components=0.95)
# print ("y: ", y.dtype, y.shape)
# lda.fit(z_data, y).transform(z_data)
# X_reduced_lda = lda.transform(z_data)
# print("lda explained variance ratio", lda.explained_variance_ratio_)


# SVD 
# pca.inverse_transform(z_data) #Transform data back, but still saved in X_reduced 
# svd = TruncatedSVD(n_components=0.95, n_iter=7, random_state=42)
# svd.fit(X.astype('int64'))
# X_reduced_svd = svd.transform(X.astype('int64'))
# print("svd explained variance", svd.explained_variance_)
# print("svd explained variance ratio", svd.explained_variance_ratio_)


# HISTOGRAM OF EXPLAINED VARIANCE 
# tot = sum(pca.explained_variance_)
# var_exp = [(i / tot)*100 for i in pca.explained_variance_]
# cum_var_exp = np.cumsum(var_exp)

# trace1 = Bar(
#         x=["PC %s" %i for i in range(1,32)],
#         y=var_exp,
#         showlegend=False)

# trace2 = Scatter(
#         x=["PC %s" %i for i in range(1,32)],
#         y=cum_var_exp,
#         name='cumulative explained variance')

# data = Data([trace1, trace2])

# layout=Layout(
#         yaxis=YAxis(title='Explained variance in percent'),
#         title='Explained variance by different principal components')

# hfig = Figure(data=data, layout=layout)
# py.plot(hfig)







pca = PCA(n_components=25)
X_r = pca.fit(z_data).transform(z_data)

lda = LinearDiscriminantAnalysis(n_components=25)
X_r2 = lda.fit(z_data, y).transform(z_data)

print("pca explained variance", pca.explained_variance_ratio_)
print("lda explained variance", lda.explained_variance_ratio_)

## VISUALIZATION
fig = plt.figure()
ax = Axes3D(fig, elev=30, azim=150)
cm_gray = plt.cm.get_cmap('gray')
cm_x = plt.cm.get_cmap('spring')
cm_z = plt.cm.get_cmap('summer')
scatter_xr = ax.scatter(X_r[:, 0], X_r[:, 1], X_r[:, 2], c=y, marker='.',
           cmap=cm_x, s=40)
scatter_xr2 = ax.scatter(X_r2[:, 0], X_r2[:, 1], X_r2[:, 2], c=y, marker='.',
           cmap=cm_z, s=40)
scatter_z = ax.scatter(z_data[:, 0], z_data[:, 1], z_data[:, 2], c=y, marker='s',
           cmap=cm_gray, s=40)
ax.set_title("Original data vs. PCA")
ax.set_xlabel("1st feature")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("2nd feature")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("3rd feature")
ax.w_zaxis.set_ticklabels([])

dfmin = np.matrix(y).min()
dfmax = np.matrix(y).max()
print("dfmin", dfmin)
print("dfmax", dfmax)


# cbx = plt.colorbar(scatter_x, ticks=None)
# cbx.set_label("PCA data")
# cbz = plt.colorbar(scatter_z, ticks=None)
# cbz.set_label("Original data")
plt.show()

















###### OLD 
# ## ORIGINAL DATA
# fig_orig = plt.figure()
# ax_orig = Axes3D(fig_orig, elev=48, azim=130)
# ax_orig.scatter(z_data[:, 0], z_data[:, 1], z_data[:, 2], c=y, marker='s',
#            cmap=plt.cm.Set1, edgecolor='k', s=40)
# ax_orig.set_title("First three data directions")
# ax_orig.set_xlabel("1st feature")
# ax_orig.w_xaxis.set_ticklabels([])
# ax_orig.set_ylabel("2nd feature")
# ax_orig.w_yaxis.set_ticklabels([])
# ax_orig.set_zlabel("3rd feature")
# ax_orig.w_zaxis.set_ticklabels([])

# ## PCA VISUALIZATION
# fig = plt.figure()
# ax = Axes3D(fig, elev=48, azim=130)
# ax.scatter(X_r[:, 0], X_r[:, 1], X_r[:, 2], c=y, marker='.',
#            cmap=plt.cm.Set1, edgecolor='k', s=40)
# ax.set_title("First three PCA directions")
# ax.set_xlabel("1st eigenvector")
# ax.w_xaxis.set_ticklabels([])
# ax.set_ylabel("2nd eigenvector")
# ax.w_yaxis.set_ticklabels([])
# ax.set_zlabel("3rd eigenvector")
# ax.w_zaxis.set_ticklabels([])

# ## LDA VISUALIZATION
# fig = plt.figure()
# ax = Axes3D(fig, elev=48, azim=130)
# ax.scatter(X_r2[:, 0], X_r2[:, 1], X_r2[:, 2], c=y, marker='.',
#            cmap=plt.cm.Set1, edgecolor='k', s=40)
# ax.set_title("First three LDA directions")
# ax.set_xlabel("1st eigenvector")
# ax.w_xaxis.set_ticklabels([])
# ax.set_ylabel("2nd eigenvector")
# ax.w_yaxis.set_ticklabels([])
# ax.set_zlabel("3rd eigenvector")
# ax.w_zaxis.set_ticklabels([])


# HEAT MAP
# sns.heatmap(np.log(pca.inverse_transform(np.eye(df.values.shape(18,25)))), cmap="hot", cbar=False)

