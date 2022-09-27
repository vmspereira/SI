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

    def __init__(self, n_components:int=2, svd:bool=True, scale:bool=True) -> None:
        """ Principal component analysis.
        :param (int) n_components: Number of components
        :param (bool) svd: Uses SVD decomposition to obtain the eigen values/vector.\
             If False, uses GEEV right eigen vector on the covariance matrix.
        :param (bool) scale: If True uses standard scaler to center and normalize the data,\
             otherwise, only centers the data values.   
        """
        self.n_components = n_components
        self.svd = svd
        self.scale = scale

    def fit(self, dataset):
        X = dataset.X
        if self.scale:
            X_scale = StandardScaler().fit_transform(dataset)
            self.X_center = X_scale.X
        else:
            # may only use the center instead of std scaler
            self.X_center = X - np.mean(X, axis=0)

        if self.svd:
            # uses SVD
            self.e_vecs, self.e_vals, vt = np.linalg.svd(self.X_center.T)
        else:
            # uses GEEV right eigen vector on the covariance matrix
            cov_matrix = np.cov(self.X_center.T)
            self.e_vals, self.e_vecs = np.linalg.eig(cov_matrix)

    def transform(self, dataset):
        self.sorted_index = np.argsort(self.e_vals)[::-1]
        self.e_vals_sorted = self.e_vals[self.sorted_index]
        self.e_vecs_sorted = self.e_vecs[:, self.sorted_index]
        self.e_vecs_subset = self.e_vecs_sorted[:, 0:self.n_components]
        X_reduced = self.e_vecs_subset.T.dot(self.X_center.T).T
        return X_reduced

    def fit_transform(self, dataset):
        self.fit(dataset)
        return self.transform(dataset)

    def variance_explained(self):
        _sum = sum(self.e_vals_sorted)
        return [(i/_sum*100) for i in self.e_vals_sorted]
