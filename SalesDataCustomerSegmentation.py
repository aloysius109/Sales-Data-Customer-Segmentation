#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 08:39:10 2024

@author: kathrynhopkins
"""
# =============================================================================
# This code shows preparatory work and visualisation for customer segmentation tasks using Principal Component Analysis (PCA) and KMeans Cluster Analysis.
# =============================================================================
#%% Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')
#%% Read in and check dataset
df = pd.read_csv('encrypted_sales_data.csv')
df.head()
df.columns.tolist()
df2 = df.groupby('Customer Name').agg({'Product Name':np.size,
                                      'Invoices.Sub Total (BCY)':[np.sum, np.size],
                                      'Total_GP':np.sum,
                                      'Invoices.Invoice Date':[np.min,np.max, np.size]})
df2.columns.tolist()
df2.columns = df2.columns.map('_'.join)
df2.columns=['Product Qty', 'Invoices Sum','Invoices Qty','Total_GP', 'Min Invoice Date', 'Max Invoice Date', 'Invoice Size']
#%% Select features
df2.columns.tolist()
features = ['Product Qty',
            'Invoices Sum',
            'Invoices Qty',
            'Total_GP']
df2[features].describe
#%% PCA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
features = features
data=df2[features]
scaler = StandardScaler()
data = scaler.fit_transform(data)
data[:,0].std()
np.save('sales_data.npy', data)
pca = PCA(n_components = 2)
res_pca = pca.fit_transform(data)
#%% Plot the components
fig, axes = plt.subplots(2,2, figsize = (15,8))
for feature, ax in zip(features, axes.ravel()):
    cmap = 'viridis'
    sizes = 20+20*data[:, features.index(feature)]
    ax.scatter(res_pca[:, 0], res_pca[:,1], s = sizes, alpha = 0.3, c = df2[feature], cmap = cmap)
    ax.tick_params(axis='both', labelleft=True, labelbottom=True, size = 0, labelsize = 8)
    ax.set_title(feature)
plt.suptitle('PCA Normalised Plots', fontsize = 12)
plt.tight_layout()
plt.savefig('PCA Plot.png')
#%% Now try kernel PCA
from sklearn.decomposition import KernelPCA
#%%
kpca = KernelPCA(n_components = 2, kernel = 'poly', degree = 2)
res_kpca_poly = kpca.fit_transform(data)
#%% And plot the result
fig, axes = plt.subplots(2,2, figsize = (16,8))
for feature, ax in zip(features, axes.ravel()):
    cmap = 'viridis'
    sizes = 20+20*data[:, features.index(feature)]
    ax.scatter(res_kpca_poly[:,0], res_kpca_poly[:,1], s = sizes, alpha = 0.3, c = df2[feature], cmap = cmap)
    ax.tick_params(axis='both', labelleft=True, labelbottom=True, size = 0, labelsize = 8)
    ax.set_title(feature)
plt.suptitle('PCA Polynomial Normalised Plots', fontsize = 12)
plt.tight_layout()
plt.savefig('PCAPolynomial.png')
#%% Now Radial Basis Function
kpca = KernelPCA(n_components = 2, kernel = 'rbf', gamma= 0.05)
res_kpca_rbf = kpca.fit_transform(data)
#%% And plot
fig, axes = plt.subplots(2,2, figsize = (16,8))
for feature, ax in zip(features, axes.ravel()):
    cmap = 'magma'
    sizes = 20+20*data[:, features.index(feature)]
    ax.scatter(res_kpca_rbf[:,0], res_kpca_rbf[:,1], s = sizes, alpha = 0.3, c = df2[feature], cmap = cmap)
    ax.tick_params(axis='both', labelleft=True, labelbottom=True, size = 0, labelsize = 8)
    ax.set_title(feature)
plt.suptitle('PCA Radial Basis Function Plots', fontsize = 12)
plt.savefig('PCARadialBasis.png')
#%% Cosine function
kpca = KernelPCA(n_components=2, kernel='cosine')
res_kpca_cos = kpca.fit_transform(data)
#%%
fig, axes = plt.subplots(2, 2, figsize=(16, 8))
for feature, ax in zip(features, axes.ravel()):
    cols = 'viridis'
    sizes = 20+20*data[:, features.index(feature)]
    ax.scatter(res_kpca_cos[:, 0], res_kpca_cos[:, 1], s=sizes, alpha=0.3, c=df2[feature], cmap=cols)
    ax.tick_params(axis='both', labelleft=True, labelbottom=True, size = 0, labelsize = 8)
    ax.set_title(feature)
plt.suptitle('PCA Cosine Function Plots', fontsize = 12)
plt.tight_layout()
plt.savefig('PCACosine.png')
#%% Append to DataFrame
df2.head()
df2['x_kpca_rbf'] = res_kpca_rbf[:, 0]
df2['y_kpca_rbf'] = res_kpca_rbf[:, 1]
df2['x_kpca_poly'] = res_kpca_poly[:, 0]
df2['y_kpca_poly'] = res_kpca_poly[:, 1]
df2['x_kpca_cos'] = res_kpca_cos[:, 0]
df2['y_kpca_cos'] = res_kpca_cos[:, 1]
df2.to_csv('data_with_latent.csv')
#%% K-Means Clustering
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
res_kpca = df2[['x_kpca_rbf', 'y_kpca_rbf']].to_numpy()
clusterer = KMeans(n_clusters=5)
clusters = clusterer.fit_predict(res_kpca)
markers = list('*hH+xXDd|.,ov^<>12348spP')
#%% And plot
fig = plt.figure(figsize = (8,5))
for cluster in np.unique(clusters):
    cluster_data = res_kpca[clusters==cluster]
    plt.scatter(cluster_data[:, 0], cluster_data[:, 1], alpha=0.7, marker=markers[cluster])
    plt.tick_params(axis='both', labelleft=True, labelbottom=True, size = 0, labelsize = 8)
    plt.title('KMeans Clusters')
    plt.show()
plt.savefig('KMeansClusters')
#%% And plot
fig, axes = plt.subplots(2, 2, figsize=(16, 8))
for feature, ax in zip(features, axes.ravel()):
    cols = 'viridis'
    for cluster in np.unique(clusters):
        sizes = 20+20*data[:, features.index(feature)][clusters==cluster]
        cluster_data = res_kpca[clusters==cluster]
        ax.scatter(cluster_data[:, 0], cluster_data[:, 1], s=sizes, alpha=0.6, cmap=cols, marker=markers[cluster], label=f'Cluster {cluster}')
    ax.tick_params(axis='both', labelleft=True, labelbottom=True, size = 0, labelsize = 8)
    ax.set_title(feature)
plt.suptitle('KMeans Cluster Plots', fontsize = 12)
plt.tight_layout()
plt.savefig('KMeansClusterPlots.png')
#%% Prepare Cluster Data
df2['cluster_kpca_cos'] = clusters
df2.to_csv('data_with_clusters.csv')
#%% Elbow method for visualising K
fig = plt.figure(figsize = (10,10))
clusterer = KMeans()
visualizer = KElbowVisualizer(clusterer, k=(2, 12), metric='distortion')
visualizer.fit(res_kpca)        
visualizer.show()
#%% Cluster Distribution
df2.groupby(['cluster_kpca_cos'])[features].mean()
df2.groupby(['cluster_kpca_cos']).count()
clusters = df2.cluster_kpca_cos
df2_normalized = df2.copy(deep=True)
# df_normalized.loc[:, features] = data
df2_normalized[features] /= df2[features].max()
df2_normalized.max()
biggest_cluster = df2.groupby(['cluster_kpca_cos']).count().max().max()
CustomerClusters = df2_normalized['cluster_kpca_cos']
CustomerClusters= pd.DataFrame(CustomerClusters)
CustomerClusters.reset_index(inplace = True)
CustomerClusters.to_csv('CustomerClusters.csv')
