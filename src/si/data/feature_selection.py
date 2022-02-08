# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Created By  : Vítor Pereira
# Created Date: 01-09-2021
# version ='0.0.1'
# ---------------------------------------------------------------------------
"""Feature Selection module"""
# ---------------------------------------------------------------------------
import numpy as np
from scipy import stats
from copy import copy
import warnings


class VarianceThreshold:

    def __init__(self, threshold=0):
        """The variance threshold is a simple baseline approach to feature selection.
           It removes all features which variance doesn't meet some threshold. By default,
           it removes all zero-variance features, i.e., features that have the same value in all samples.

        :param threshold: The non negative threshold value, defaults to 0.
        :type threshold: int, optional
        """
        if threshold < 0:
            warnings.warn("The thershold must be a non-negative value.")
        self.threshold = threshold

    def fit(self, dataset):
        X = dataset.X
        self._var = np.var(X, axis=0)

    def transform(self, dataset, inline=False):
        X = dataset.X
        cond = self._var > self.threshold
        idxs = [i for i in range(len(cond)) if cond[i]]
        X_trans = X[:, idxs]
        xnames = [dataset._xnames[i] for i in idxs]
        if inline:
            dataset.X = X_trans
            dataset._xnames = xnames
            return dataset
        else:
            from .dataset import Dataset
            return Dataset(copy(X_trans),
                           copy(dataset.y),
                           xnames,
                           copy(dataset._yname)
                           )

    def fit_transform(self, dataset, inline=False):
        self.fit(dataset)
        return self.transform(dataset, inline=inline)


def f_classif(dataset):
    """Scoring function for classifications.
    Compute the ANOVA F-value for the provided sample.


    :param dataset: A labeled dataset
    :type dataset: Dataset
    :return: F scores and p-values
    :rtype: a tupple of np.arrays
    """
    X = dataset.X
    y = dataset.y
    # selectiona os registos por class
    args = [X[y == a] for a in np.unique(y)]
    # Calcula as F-statistics e p values.
    # Queremos identificar se existe uma differença significativa
    # entre os tratamentos (features)
    F, p = stats.f_oneway(*args)
    return F, p


def f_regress(dataset):
    """Scoring function for regressions

    :param dataset: A labeled dataset
    :type dataset: Dataset
    :return: F scores and p-values
    :rtype: a tupple of np.arrays
    """
    X = dataset.X
    y = dataset.y
    correlation_coefficient = np.array([stats.pearsonr(X[:, i], y)[0] for i in range(X.shape[1])])
    deg_of_freedom = y.size - 2
    corr_coef_squared = correlation_coefficient ** 2
    F = corr_coef_squared / (1 - corr_coef_squared) * deg_of_freedom
    p = stats.f.sf(F, 1, deg_of_freedom)
    return F, p


class SelectKBest:

    def __init__(self, k: int, score_func=f_classif):
        """[summary]

        :param k: [description]
        :type k: int
        :param score_func: [description], defaults to f_classif
        :type score_func: [type], optional
        """
        self.k = k
        self.score_func = score_func

    def fit(self, dataset):
        self.F, self.p = self.score_func(dataset)

    def transform(self, dataset, inline=False):
        # Nota que os p e F têm uma relação inversa, qto maior
        # o valor de F menor o p.
        # Maiores valores de F correspondem a uma rejeição com probabilidade
        # (1-p) da hipótese nula.
        idxs = self.F.argsort()[-self.k:]
        idxs.sort()
        X_trans = dataset.X[:, idxs.tolist()]
        xnames = [dataset._xnames[i] for i in idxs.tolist()]
        if inline:
            dataset.X = X_trans
            dataset._xnames = xnames
            return dataset
        else:
            from .dataset import Dataset
            return Dataset(copy(X_trans),
                           copy(dataset.y),
                           xnames,
                           copy(dataset._yname)
                           )

    def fit_transform(self, dataset, inline=False):
        self.fit(dataset)
        return self.transform(dataset, inline=inline)
