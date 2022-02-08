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

    def __init__(self, models, fvote, score):
        super().__init__()
        self.models = models
        self.fvote = fvote
        self.score = score

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
