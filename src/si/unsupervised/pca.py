# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Created By  : Vítor Pereira
# Created Date: 01-09-2021
# version ='0.0.1'
# ---------------------------------------------------------------------------
"""Principal Component Analysis module"""
# ---------------------------------------------------------------------------
import numpy as np
from ..data import StandardScaler


class PCA:

    def __init__(self, n_components:int=2, svd:bool=True, scale_data:bool=True) -> None:
        """ Principal component analysis.
        :param (int) n_components: Number of components
        :param (bool) svd: Uses SVD decomposition to obtain the eigen values/vector.\
             If False, uses GEEV right eigen vector on the covariance matrix.
        :param (bool) scale: If True uses standard scaler to center and normalize the data,\
             otherwise, only centers the data values.

        --------------------------
        The principal components are computed using eigen vectores (x) and values (λ), solutions
        of the equation:
          A x =  λ x
        where A is the data covariance matrix.

        The covariance matrix A tells how much the variables differ from one another, 
        and we want to preserve the directions along which there is more variability.
        These directions are the eigen vectores (x) with higher eigen values (λ).
        The eigen values are a measure of how spread is the data along the corresponding
        eigen vector. 

        PCA has some assumptions: 
            - There must be linearity in the data set, i.e.,
              the variables combine in a linear manner to form the dataset. 
        
            - The variables exhibit relationships among themselves.

        Some rule of thumb:

            - The number of observations should be at least 150 with a ratio measurement
              of 5:1.

            - Extreme values that deviate from other data points in any dataset, outliers, 
              should be preferebly removed. Usually values outside the (mean ± 3*std) are 
              considered outliers.    
        """
        self.n_components = n_components
        self.svd = svd
        self.scale_data = scale_data

    def scale(self, dataset):
        X = dataset.X
        if self.scale_data:
            X_scale = StandardScaler().fit_transform(dataset)
            X_center = X_scale.X
        else:
            # Centers only instead of std scaler
            X_center = X - np.mean(X, axis=0)
        return X_center


    def fit(self, dataset):
        """Computes the eigen values and vectors"""
        self.X_center = self.scale(dataset)
        if self.svd:
            # uses SVD
            # SVD factorizes a matric into a product of 3 other matrices:
            # A = U S V*, where U and V are ortogonal and S is diagonal.
            # U is a matrix of eigen vectores, while the diagonal of S are
            # eigen values.
            self.e_vecs, self.e_vals, vt = np.linalg.svd(self.X_center.T)
        else:
            # uses GEEV right eigen vector on the covariance matrix
            cov_matrix = np.cov(self.X_center.T)
            self.e_vals, self.e_vecs = np.linalg.eig(cov_matrix)

    def transform(self, dataset):
        """
        The principal components, eigen vectors,
        are used to build a transition matrix from an higher
        to a lower dimension.
        """ 
        X_center = self.scale(dataset) 
        self.sorted_index = np.argsort(self.e_vals)[::-1]
        self.e_vals_sorted = self.e_vals[self.sorted_index]
        self.e_vecs_sorted = self.e_vecs[:, self.sorted_index]
        # transition matrix, or change of base matrix. 
        self.e_vecs_subset = self.e_vecs_sorted[:, 0:self.n_components]
        # projects the data into a lower dimension.
        X_reduced = self.e_vecs_subset.T.dot(X_center.T).T
        return X_reduced

    def fit_transform(self, dataset):
        self.fit(dataset)
        return self.transform(dataset)

    def variance_explained(self):
        _sum = sum(self.e_vals_sorted)
        return [(i/_sum*100) for i in self.e_vals_sorted]
