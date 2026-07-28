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

    # `predict` takes a 2-D X (n_samples, n_features) and returns one
    # prediction per row -- see the note on Model.predicts_batch.
    predicts_batch = True

    def __init__(self, alpha=1.0):
        """
        Naive Bayesian for categorical data.
        Bayesian inference calculates the posterior probability as a consequence of two antecedents
        a prior probability and a "likelihood function" derived from a statistical model of
        observed data.

        :param float alpha: an additive term used to ensure no null likelihood. Default 1.0.

        ---------------------------------------------------------------
        Bayes' theorem: (conditional probabilities)

        P(Y|X) = P(X|Y) * P(Y) / P(X)
        [posterior = likelihood * prior / evidence]

        For given a set of feature values X=x1,x2,...,xn we want to know the probability
        of each class y. It is assumed that X and Y are both possible and dependent (the
        class values depend on the features), while a strong (naive) independence is
        assumed between the features, that is,

            P(X= x1,x2,...,xn) = P(X=x1) P(X=x2) ... P(X=xn)

        and

            P(X= x1,x2,...,xn|Y=y) = P(X=x1|Y=y) P(X=x2|Y=y) ... P(X=xn|Y=y)

        This enables calculating the likelihoods and evidences.

        When to use it: a fast, simple, surprisingly strong baseline for
        high-dimensional categorical/count data such as text (bag-of-words). The
        naive independence assumption is usually false, yet the classifier often
        works well because we only need the correct class to win the argmax, not
        the probabilities themselves to be exact.

        About `alpha` (Laplace / additive smoothing): if a feature value is
        never observed for a class in training, its raw likelihood is 0, which
        would zero out the whole product P(x1|y)*...*P(xn|y). Adding alpha to
        every count guarantees strictly positive likelihoods.
        """
        super().__init__()
        # prior[c]   = P(Y = c), estimated from class frequencies.
        # lk[c, j]   = P(feature j | Y = c), the per-class likelihood table.
        self.prior = None
        self.lk = None
        self.alpha = alpha

    def fit(self, dataset):
        X, y = dataset.getXy()
        self.dataset = dataset
        n = X.shape[0]

        # The labels seen during training, in sorted order. Every per-class
        # array below is aligned with this list, so a position in `prior` / `lk`
        # (and hence an argmax over the posteriors) maps back to a label through
        # self.classes. Stored rather than recomputed so predict can do that
        # mapping.
        self.classes = np.unique(y)

        # one (n_samples_c, n_features) array per class; kept as a list because
        # classes may have different sample counts (a ragged np.array is invalid)
        X_by_class = [X[y == c] for c in self.classes]
        # Prior P(Y=c) = (#samples in class c) / (total #samples).
        self.prior = np.array([len(X_class) / n for X_class in X_by_class])

        # counts[c, j] = total occurrences of feature j across class c's samples
        # (+ alpha smoothing so no count is 0). Shape: (n_classes, n_features).
        counts = np.array([sub_arr.sum(axis=0) for sub_arr in X_by_class]) + self.alpha
        # Normalise each class row so the per-feature likelihoods of that class
        # sum to 1 -> lk[c, j] = P(feature j | Y = c). The reshape makes the
        # per-class totals a column vector so the division broadcasts row-wise.
        self.lk = counts / counts.sum(axis=1).reshape(-1, 1)
        self.is_fitted = True

    def predict_proba(self, x):
        """ Predict probability of class membership.

        For each sample and each class c we form the (unnormalised) posterior
        numerator P(Y=c) * prod_j P(feature j | Y=c), then divide by the sum
        over classes (the evidence P(X)) so the posteriors sum to 1. The
        evidence is the same for all classes, so it does not change the argmax.
        """

        assert self.is_fitted, 'Model must be fit before predicting'

        # class_numerators[i, c] will hold the posterior numerator of sample i
        # for class c. Shape: (n_samples, n_classes).
        class_numerators = np.zeros(shape=(x.shape[0], self.prior.shape[0]))
        for i, x in enumerate(x):
            # Boolean mask of which features are present (non-zero) in sample i.
            exists = x.astype(bool)
            # Likelihood of each present feature, raised to its count x[j] (so a
            # feature seen k times contributes P(.|c)^k). Shape: (n_classes, n_present).
            lk_present = self.lk[:, exists] ** x[exists]
            # Naive independence: multiply the per-feature likelihoods together
            # -> P(x | Y=c) for every class. Shape: (n_classes,).
            lk_marginal = (lk_present).prod(axis=1)
            # Posterior numerator = likelihood * prior, per class.
            class_numerators[i] = lk_marginal * self.prior

        # Evidence P(X): sum the numerators over classes (per sample). reshape
        # to a column so the division normalises each row to sum to 1.
        normalize_term = class_numerators.sum(axis=1).reshape(-1, 1)
        conditional_probas = class_numerators / normalize_term
        assert (conditional_probas.sum(axis=1) - 1 < 0.001).all(), 'Rows should sum to 1'
        return conditional_probas

    def predict(self, x):
        assert self.is_fitted, 'Model must be fit before predicting'
        # Predicted class = the one with the highest posterior probability (MAP).
        #
        # argmax gives a COLUMN POSITION in the posterior matrix, which is an
        # index into self.classes -- not a label. Returning it directly (as this
        # used to) is only correct when the labels are 0..k-1; with labels
        # {1, 2} or {'a', 'b'} cost() compared positions against labels and
        # reported 0% accuracy. Indexing self.classes maps it back.
        positions = self.predict_proba(x).argmax(axis=1)
        return self.classes[positions]

    def cost(self, X=None, y=None):
        assert self.is_fitted, 'Model must be fit before predicting'
        X = X if X is not None else self.dataset.X
        y = y if y is not None else self.dataset.y

        # NaiveBayes.predict is a *batch* predictor (it operates on a 2D X and
        # returns one label per row), so call it directly on X. The
        # apply_along_axis bridge is only for the single-sample predictors
        # (KNN, LinearRegression, ...); feeding a 1D column to predict here
        # would break predict_proba's broadcasting.
        return accuracy_score(y, self.predict(X))
