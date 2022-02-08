# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Created By  : Vítor Pereira
# Created Date: 01-09-2021
# version ='0.0.1'
# ---------------------------------------------------------------------------
"""Naive Bayesian module"""
# ---------------------------------------------------------------------------

from .model import Model
from ..util import accuracy_score
import numpy as np


class NaiveBayes(Model):

    def __init__(self, alpha=1.0):
        super(NaiveBayes).__init__()
        self.prior = None
        self.lk = None
        # alpha is an additive term used to ensure no null likelihood
        self.alpha = alpha

    def fit(self, dataset):
        """Bayesian inference derives the posterior probability as a consequence of two antecedents
        a prior probability and a "likelihood function" derived from a statistical model for the
        observed data.

        Bayes' theorem:

        P(y|x) = P(x|y) * P(y) / p(x)
        [posterior = likelihood * prior / evidence]
        """
        X, y = dataset.getXy()
        self.dataset = dataset
        n = X.shape[0]

        X_by_class = np.array([X[y == c] for c in np.unique(y)])
        self.prior = np.array([len(X_class) / n for X_class in X_by_class])

        counts = np.array([sub_arr.sum(axis=0) for sub_arr in X_by_class]) + self.alpha
        self.lk = self.counts / counts.sum(axis=1).reshape(-1, 1)
        self.is_fitted = True

    def predict_proba(self, x):
        """ Predict probability of class membership """

        assert self.is_fitted, 'Model must be fit before predicting'

        # loop over each observation to calculate conditional probabilities
        class_numerators = np.zeros(shape=(x.shape[0], self.prior.shape[0]))
        for i, x in enumerate(x):
            exists = x.astype(bool)
            lk_present = self.lk[:, exists] ** x[exists]
            lk_marginal = (lk_present).prod(axis=1)
            class_numerators[i] = lk_marginal * self.prior

        normalize_term = class_numerators.sum(axis=1).reshape(-1, 1)
        conditional_probas = class_numerators / normalize_term
        assert (conditional_probas.sum(axis=1) - 1 < 0.001).all(), 'Rows should sum to 1'
        return conditional_probas

    def predict(self, x):
        assert self.is_fitted, 'Model must be fit before predicting'
        return self.predict_proba(x).argmax(axis=1)

    def cost(self, X=None, y=None):
        assert self.is_fitted, 'Model must be fit before predicting'
        X = X if X is not None else self.dataset.X
        y = y if y is not None else self.dataset.y

        y_pred = np.ma.apply_along_axis(self.predict,
                                        axis=0, arr=X.T)
        return accuracy_score(y, y_pred)
