# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Created By  : Vítor Pereira
# Created Date: 01-09-2021
# version ='0.0.1'
# ---------------------------------------------------------------------------
"""Standard Scaler module"""
# ---------------------------------------------------------------------------
import numpy as np
from copy import copy
from .transformer import Transformer


class StandardScaler(Transformer):
    """
    Standardize features by centering the mean to 0 and unit variance.
    The standard score (z-score) of an instance is calculated by:
        z = (x - u) / s
    where u is the mean of the training data and s is the standard deviation.

    After this transform every feature has mean 0 and standard deviation 1, so
    all features live on the same scale and become directly comparable.

    Why standardize before training?
    --------------------------------
    Many algorithms are sensitive to the scale of the inputs:
      * Distance-based methods (KNN, K-means) and PCA would otherwise let a
        feature with a large numeric range dominate purely because of its
        units, not its importance.
      * Gradient-based optimizers converge faster and more reliably when all
        features have similar scales (it avoids exploding/vanishing gradients
        and ill-conditioned, elongated loss surfaces).
    Note we LEARN u and s on the training data (in `fit`) and reuse the SAME
    values to transform any future data, so train and test are scaled
    consistently.

    Attributes
    ----------
    _mean : numpy array of shape (n_features, )
        The mean of each feature in the training set.
    _var : numpy array of shape (n_features, )
        The variance of each feature in the training set.
    _scale : numpy array of shape (n_features, )
        The divisor actually used: sqrt(_var), except 1 wherever the variance is
        zero so a constant feature does not divide by zero.
    """

    def fit(self, dataset):
        """
        Calculate and store the mean and variance of each feature in the
        training set.
        Parameters
        ----------
        dataset : A Dataset object to be standardized
        """
        # axis=0 -> compute statistics down the rows, i.e. one value PER
        # FEATURE (column). Both arrays have shape (n_features,).
        X = dataset.X
        self._mean = np.mean(X, axis=0)
        self._var = np.var(X, axis=0)
        # The scale actually divided by. A CONSTANT feature has zero variance,
        # so dividing by sqrt(var) produced NaN for that whole column and every
        # model downstream silently trained on NaNs. A constant feature carries
        # no information to rescale: centring already maps it to 0, so the scale
        # is set to 1 and the column simply becomes zeros.
        self._scale = np.sqrt(self._var)
        self._scale = np.where(self._scale == 0, 1.0, self._scale)
        return self

    def transform(self, dataset, inline=False):
        """
        Standardize data by subtracting out the mean and dividing by
        standard deviation calculated during fitting.
        Parameters
        ----------
        dataset : A Dataset object to be standardized
        Returns
        -------
        A Dataset object with standardized data.
        """
        # Apply z = (x - u) / s. np.sqrt(var) gives the standard deviation s.
        # The subtraction and division broadcast the per-feature mean/std
        # across every row, so each column is standardized independently.
        assert hasattr(self, '_scale'), \
            'StandardScaler must be fit before transforming'
        X = dataset.X
        Z = (X - self._mean) / self._scale

        if inline:
            # Modify the dataset in place and return it.
            dataset.X = Z
            return dataset
        else:
            # Build and return a brand new Dataset, leaving the input untouched.
            from . import Dataset
            return Dataset(Z,
                           copy(dataset.y),
                           copy(dataset._xnames),
                           copy(dataset._yname))

    def inverse_transform(self, dataset, inline=False):
        """
        Transform data back into orginal state by multiplying by standard
        deviation and adding the mean back in.
        Inverse standard scaler:
            x = z * s + u
        where s is the standard deviation, and u is the mean.

        This simply undoes the z-score: it is the algebraic inverse of
        z = (x - u) / s, recovering the data in its original units (useful for
        interpreting results or reporting predictions on the original scale).
        Parameters
        ----------
        dataset : A standardized Dataset object
        Returns
        -------
        Dataset object
        """
        # x = z * s + u, reusing the s and u learned during fit. The same
        # guarded scale is used, so the round trip is exact for ordinary
        # features; a constant feature is recovered from its mean alone (its
        # spread was zero, so nothing was lost).
        assert hasattr(self, '_scale'), \
            'StandardScaler must be fit before inverse transforming'
        X = dataset.X * self._scale + self._mean
        if inline:
            dataset.X = X
            return dataset
        else:
            from . import Dataset
            return Dataset(X,
                           copy(dataset.y),
                           copy(dataset._xnames),
                           copy(dataset._yname))
