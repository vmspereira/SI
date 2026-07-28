# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Created By  : Vítor Pereira
# Created Date: 01-09-2021
# version ='0.0.1'
# ---------------------------------------------------------------------------
"""Defines an interface for supervised learning models"""
# ---------------------------------------------------------------------------

from abc import ABC, abstractmethod


class Model(ABC):

    # Does `predict` expect a batch of samples, or a single one?
    #
    # This library is deliberately inconsistent about it. `KNN`,
    # `LinearRegression`, `LogisticRegression` and `DecisionTree` predict ONE
    # sample: they take a 1-D `x` (n_features,) and return a single value.
    # `RandomForest`, `NaiveBayes`, `LDA`, `SVM` and `NN` predict a BATCH: they
    # take a 2-D `X` (n_samples, n_features) and return one prediction per row.
    #
    # Wrappers that call an arbitrary model (`Ensemble`, and `cost` via
    # `np.ma.apply_along_axis`) cannot tell these apart by inspecting `x` alone
    # — a 1-D array is a valid single sample AND a valid one-feature batch. So
    # instead of guessing, every model declares its convention here and the
    # wrapper adapts explicitly.
    predicts_batch = False

    def __init__(self):
        """ Abstract class defining an interface for
        supervised learning models.

        A model needs to implement a `fit`, a `predict` and a `cost` method.
        A model that predicts whole batches must also set the class attribute
        `predicts_batch = True`.
        """
        self.is_fitted = False

    @abstractmethod
    def fit(self, dataset, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def predict(self, x):
        raise NotImplementedError

    @abstractmethod
    def cost(self, *args, **kwarg):
        raise NotImplementedError
