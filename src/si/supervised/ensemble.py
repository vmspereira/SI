# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Created By  : Vítor Pereira
# Created Date: 01-09-2021
# version ='0.0.1'
# ---------------------------------------------------------------------------
"""Ensemble module"""
# ---------------------------------------------------------------------------
from .model import Model
from collections import Counter
import numpy as np


def majority(values):
    """Hard vote: the value that the members chose most often.

    Used for classification. Ties are broken towards the smallest label so
    that the vote is reproducible; `max(set(values), key=values.count)` would
    resolve a tie by `set` iteration order, which is an implementation detail.
    """
    counts = Counter(values)
    most = max(counts.values())
    return min(v for v, c in counts.items() if c == most)


def average(values):
    """Soft vote: the mean of the members' predictions (for regression)."""
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
        """Combine the members' predictions for the single sample `x`."""
        assert self.is_fitted, 'Model must be fit before predicting'
        # One prediction per member, each obtained in that member's own
        # calling convention, then reduced to a single answer by the vote.
        preds = [self._predict_member(model, x) for model in self.models]
        vote = self.fvote(preds)
        return vote

    @staticmethod
    def _predict_member(model, x):
        """Ask one member model to predict the single sample `x`.

        Members disagree on how they want to be called (see the note on
        `Model.predicts_batch`), so translate in both directions:

        - a batch member wants a 2-D (n_samples, n_features) matrix and answers
          with one prediction per row, so present `x` as a single-row matrix
          and unwrap the length-1 result;
        - a single-sample member takes the 1-D `x` unchanged.

        Without this translation, handing a 1-D `x` straight to a batch member
        makes it iterate over *features* instead of samples — which is how
        `Ensemble([NaiveBayes(), LDA()])` used to fail with an unhelpful
        broadcasting error.
        """
        if getattr(model, 'predicts_batch', False):
            return model.predict(np.asarray(x).reshape(1, -1))[0]
        return model.predict(x)

    def cost(self, X=None, y=None):
        assert self.is_fitted, 'Model must first be fit'
        X = X if X is not None else self.dataset.X
        y = y if y is not None else self.dataset.y
        y_pred = np.ma.apply_along_axis(self.predict,
                                        axis=0, arr=X.T)
        return self.score(y, y_pred)
