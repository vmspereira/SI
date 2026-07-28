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

    def __init__(self,
                 n_components: int = 2,
                 svd: bool = True,
                 scale_data: bool = True) -> None:
        """
        Principal component analysis.

        :param (int) n_components: Number of components
        :param (bool) svd: Uses SVD decomposition to obtain the eigen values/vector.\
             If False, uses GEEV right eigen vector on the covariance matrix.
        :param (bool) scale: If True uses standard scaler to center and normalize the data,\
             otherwise, only centers the data values.

        --------------------------
        What is PCA trying to do?
        -------------------------
        PCA finds a new set of axes (the "principal components") for the data
        such that the first axis points in the direction of greatest variance,
        the second in the direction of greatest REMAINING variance orthogonal
        to the first, and so on. Projecting the data onto the first few of
        these axes gives a lower-dimensional representation that keeps as much
        of the original spread (information) as possible. This is why PCA is
        used for dimensionality reduction, visualization and de-correlation.

        The principal components are computed using eigen vectores (x) and values (λ), solutions
        of the equation:
          A x =  λ x
        where A is the data covariance matrix.

        The covariance matrix A tells how much the variables differ from one another,
        and we want to preserve the directions along which there is more variability.
        These directions are the eigen vectores (x) with higher eigen values (λ).
        The eigen values are a measure of how spread is the data along the corresponding
        eigen vector.

        Why eigenvectors of the covariance matrix?
        ------------------------------------------
        The variance of the data projected onto a unit direction v is v^T A v.
        Maximizing that quantity (subject to ||v|| = 1) is a classic Lagrange
        problem whose solution is A v = λ v -- i.e. v must be an eigenvector of
        A, and the variance captured equals its eigenvalue λ. So "keep the
        directions of highest variance" is literally "keep the eigenvectors
        with the largest eigenvalues".

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
        # PCA is about variance, and variance is measured RELATIVE TO THE MEAN,
        # so the data must always be centred (mean of every feature set to 0)
        # before computing the covariance/SVD. Optionally we also divide by the
        # standard deviation: without that, a feature measured in large units
        # (e.g. salary in dollars) would dominate the variance purely because
        # of its scale, not because it carries more information.
        X = dataset.X
        if self.scale_data:
            # Full z-score standardization (centre AND unit variance).
            X_scale = StandardScaler().fit_transform(dataset)
            X_center = X_scale.X
        else:
            # Centers only instead of std scaler
            X_center = X - np.mean(X, axis=0)
        return X_center

    def fit(self, dataset):
        """Computes the eigen values and vectors.

        Two equivalent paths to the principal components are offered:
          * svd=True  -> Singular Value Decomposition of the (centred) data.
          * svd=False -> eigendecomposition of the covariance matrix.
        Both yield the same eigenvectors/eigenvalues (up to ordering and sign);
        SVD is generally more numerically stable and is the route real
        libraries take.
        """
        self.X_center = self.scale(dataset)
        if self.svd:
            # uses SVD
            # SVD factorizes a matrix into a product of 3 other matrices:
            # A = U S V*, where U and V are orthogonal and S is diagonal.
            # U holds the eigen vectors; the diagonal of S holds the singular
            # values. The variance along each component is proportional to the
            # SQUARE of the singular value, so square them to obtain quantities
            # comparable to the covariance-matrix eigenvalues (this keeps the
            # ordering identical and makes variance_explained correct).
            self.e_vecs, singular_values, vt = np.linalg.svd(self.X_center.T)
            self.e_vals = singular_values ** 2
        else:
            # uses GEEV right eigen vector on the covariance matrix
            # np.cov expects variables in rows, observations in columns, hence
            # the transpose: cov_matrix has shape (n_features, n_features) and
            # entry (i, j) is the covariance between features i and j.
            cov_matrix = np.cov(self.X_center.T)
            # np.linalg.eig solves A x = λ x directly: e_vals holds the
            # eigenvalues (variance along each component) and the COLUMNS of
            # e_vecs hold the corresponding eigenvectors (the components).
            self.e_vals, self.e_vecs = np.linalg.eig(cov_matrix)

    def transform(self, dataset):
        """
        The principal components, eigen vectors,
        are used to build a transition matrix from an higher
        to a lower dimension.
        """
        X_center = self.scale(dataset)
        # Eigenvalues are not guaranteed to come out sorted, so order them from
        # largest to smallest variance ([::-1] reverses the ascending argsort).
        self.sorted_index = np.argsort(self.e_vals)[::-1]
        self.e_vals_sorted = self.e_vals[self.sorted_index]
        # Reorder the eigenvectors (columns) to match the sorted eigenvalues.
        self.e_vecs_sorted = self.e_vecs[:, self.sorted_index]
        # transition matrix, or change of base matrix.
        # Keep only the top n_components directions -> shape
        # (n_features, n_components). These columns are the new axes.
        self.e_vecs_subset = self.e_vecs_sorted[:, 0:self.n_components]
        # projects the data into a lower dimension.
        # For each point we take its dot product with each kept component.
        # (components^T . X^T)^T gives a result of shape
        # (n_samples, n_components): the coordinates of every point in the new,
        # reduced basis.
        X_reduced = self.e_vecs_subset.T.dot(X_center.T).T
        return X_reduced

    def fit_transform(self, dataset):
        # Convenience: learn the components (fit) then project the data
        # (transform) in one call.
        self.fit(dataset)
        return self.transform(dataset)

    def variance_explained(self):
        # "Variance explained" = the share of the data's total variance that
        # each component accounts for. Since each eigenvalue IS the variance
        # along its component, the fraction explained by component i is simply
        # eigenvalue_i / sum(all eigenvalues). Multiplying by 100 gives a
        # percentage. These values help decide how many components to keep
        # (e.g. enough to reach 95% of the total variance).
        _sum = sum(self.e_vals_sorted)
        return [(i / _sum * 100) for i in self.e_vals_sorted]
