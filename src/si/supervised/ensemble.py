# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Created By  : Vítor Pereira
# Created Date: 01-09-2021
# version ='0.0.1'
# ---------------------------------------------------------------------------
"""Ensemble module"""
# ---------------------------------------------------------------------------
from .model import Model
import numpy as np


def majority(values):
    return max(set(values), key=values.count)


def average(values):
    return sum(values)/len(values)


class Ensemble(Model):

    def __init__(self, models, score, fvote=majority, fitted=False):
        """Bagging Model Ensemble

        Args:
            models (list[Model]): a list of models.   
            score (callable): the scoring function.
            fvote (callable, optional): the decision making function (average,majority).
               Default to majority. 
            fitted (bool, optional): If the models were previously trained. Defaults to False.
            
        Note: majority should be used for classifications tasks while
              average for regression tasks.
        """
        super().__init__()
        self.models = models
        self.fvote = fvote
        self.score = score
        self.is_fitted = fitted
        

    def fit(self, dataset):
        self.dataset = dataset
        for model in self.models:
            model.fit(dataset)
        self.is_fitted = True

    def predict(self, x):
        assert self.is_fitted, 'Model must be fit before predicting'
        preds = [model.predict(x) for model in self.models]
        vote = self.fvote(preds)
        return vote

    def cost(self, X=None, y=None):
        assert self.is_fitted, 'Model must first be fit'
        X = X if X is not None else self.dataset.X
        y = y if y is not None else self.dataset.y
        y_pred = np.ma.apply_along_axis(self.predict,
                                        axis=0, arr=X.T)
        return self.score(y, y_pred)
