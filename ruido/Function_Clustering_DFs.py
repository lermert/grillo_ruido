# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 11:57:54 2020

@author: Loic Viens
"""

import numpy as np
from sklearn.decomposition import PCA 
from sklearn.mixture import GaussianMixture
from kneed import KneeLocator


def Clustering_PCA_GMM(mat, range_GMM, n_pca_min=3, min_cumul_var_perc=0.9):
    """
    Inputs:
        mat: Matrix of correlation functions
        PC_nb: number of principal components to keep (typically 5 - 15)
        range_GMM: range of number of clusters to determine the best number of clusters using the knee (or elbow) method (typically 2 - 15)
    Outputs:
        pca_output: Output of the PCA
        var: Cumulative explained variance for each of from 1 to "PC_nb"
        models: GMM for different number of clusters
        n_clusters: Best number of clusters determined by the knee method
        gmixfinPCA: Clustering of the data with "n_clusters"
        probs: Probability that the data belong to the cluster they were assigned to.
        BICF: BIC score for the different number of clusters

    """
    # Perform PCA with the number of principal components given in input

    cumul_var_perc = 0.
    n_pca = n_pca_min
    while cumul_var_perc < min_cumul_var_perc:
        pca = PCA(n_components=n_pca)
        pca_output = pca.fit_transform(mat)
        cumul_var_perc = np.cumsum(pca.explained_variance_ratio_)[-1] # Cumulative variance
        n_pca += 1

    print('The first ' + str(n_pca-1) + ' PCs explain ' + str(cumul_var_perc) + ' % of the cumulative variance')
    
    # Compute the GMM for different number of clusters set by "range_GMM"
    models = [GaussianMixture(n, covariance_type='full', random_state=0, max_iter=10000).fit(pca_output) for n in range_GMM]
    # Compute the Bayesian information criterion (BIC) for each model
    BICF =[m.bic(pca_output)/1000 for m in models]
     # Determine the best number of clusters using the knee (or elbow) method from the BIC scores
    kn = KneeLocator(range_GMM, BICF, S=1, curve='convex', direction='decreasing')
    n_clusters = kn.knee
    if n_clusters == None:
        n_clusters = 1
    
    # Perform clustering for the best number of clusters
    gmix = GaussianMixture(n_components=n_clusters, covariance_type='full', max_iter=10000)
    gmix.fit(pca_output)
    gmixfinPCA = gmix.predict(pca_output)
    probs = gmix.predict_proba(pca_output)
    
    return pca_output, cumul_var_perc, models, n_clusters, gmixfinPCA, probs, BICF


def run_pca(mat, n_pca_min=3, min_cumul_var_perc=0.9):
    """
    Inputs:
        mat: Matrix of correlation functions
        PC_nb: number of principal components to keep (typically 5 - 15)
        range_GMM: range of number of clusters to determine the best number of clusters using the knee (or elbow) method (typically 2 - 15)
    Outputs:
        pca_output: Output of the PCA
        var: Cumulative explained variance for each of from 1 to "PC_nb"
        models: GMM for different number of clusters
        n_clusters: Best number of clusters determined by the knee method
        gmixfinPCA: Clustering of the data with "n_clusters"
        probs: Probability that the data belong to the cluster they were assigned to.
        BICF: BIC score for the different number of clusters

    """
    # Perform PCA with the number of principal components given in input

    cumul_var_perc = 0.
    n_pca = n_pca_min
    while cumul_var_perc < min_cumul_var_perc:
        pca = PCA(n_components=n_pca)
        pca = pca.fit(mat)
        # Cumulative variance
        cumul_var_perc = np.cumsum(pca.explained_variance_ratio_)[-1]
        n_pca += 1

    print('The first ' + str(n_pca-1) + ' PCs explain ' + str(cumul_var_perc) +
          ' % of the cumulative variance')

    return(pca)


def gmm(matpc, range_GMM):

    # Compute the GMM for different number of clusters set by "range_GMM"
    models = [GaussianMixture(n, covariance_type='full', random_state=0,
                              max_iter=10000).fit(matpc) for n in range_GMM]
    # Compute the Bayesian information criterion (BIC) for each model
    BICF = [m.bic(matpc) / 1000 for m in models]
    # Determine the best number of clusters using the knee (or elbow)
    kn = KneeLocator(range_GMM, BICF, S=1, curve='convex',
                     direction='decreasing')
    n_clusters = kn.knee
    if n_clusters is None:
        n_clusters = 1

    # Perform clustering for the best number of clusters
    gmix = GaussianMixture(n_components=n_clusters, covariance_type='full',
                           max_iter=10000)
    gmix.fit(matpc)
    gmixfinPCA = gmix.predict(matpc)
    probs = gmix.predict_proba(matpc)

    return models, n_clusters, gmixfinPCA, probs, BICF
