# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Created By  : Vítor Pereira
# Created Date: 01-09-2021
# version ='0.0.1'
# ---------------------------------------------------------------------------
"""Cross Validation module"""
# ---------------------------------------------------------------------------

from .helpers import train_test_split, predict_all
import numpy as np
import itertools
import copy


class CrossValidationScore:

    def __init__(self, model, dataset, score=None, ** kwargs):
        """Estimates how well a model generalises, by scoring it on data it was
        not trained on.

        Two resampling strategies are available:

        `strategy='kfold'` (the default) is k-fold cross-validation proper. The
        samples are shuffled once and cut into `cv` DISJOINT folds; each fold
        serves as the test set exactly once while the other k-1 folds train the
        model. Every sample is therefore tested on exactly once, and the k test
        scores partition the dataset rather than overlapping.

        `strategy='holdout'` is repeated random subsampling (Monte Carlo
        cross-validation): draw an independent random `split` train/test
        partition `cv` times over. It is a legitimate method, but the rounds are
        independent, so a sample may land in the test set several times or never
        -- the test sets overlap and coverage is not guaranteed. This was the
        original behaviour of this class, kept for the notebooks that rely on
        it.

        :param Model model: the model to evaluate.
        :param Dataset dataset: the data.
        :param callable score: scoring function score(y_true, y_pred). If None,
            the model's own `cost` is used instead.
        :param int cv: number of folds (kfold) or rounds (holdout). Default 3.
        :param float split: train fraction, used by `holdout` ONLY. Default 0.8.
        :param str strategy: 'kfold' or 'holdout'. Default 'kfold'.
        :param int random_state: seed for the shuffle, for reproducible folds.
        """
        self.model = model
        self.dataset = dataset
        self.score = score
        self.cv = kwargs.get('cv', 3)
        self.split = kwargs.get('split', 0.8)
        self.strategy = kwargs.get('strategy', 'kfold')
        self.random_state = kwargs.get('random_state', None)
        if self.strategy not in ('kfold', 'holdout'):
            raise ValueError(
                f"Unknown strategy '{self.strategy}': use 'kfold' or 'holdout'.")
        self.train_scores = None
        self.test_scores = None
        self.ds = None

    def _subset(self, indices):
        """Builds a Dataset from the rows of `self.dataset` at `indices`."""
        from ..data import Dataset
        return Dataset(self.dataset.X[indices],
                       self.dataset.y[indices],
                       self.dataset._xnames,
                       self.dataset._yname)

    def _splits(self):
        """Yields the (train, test) pair for each round of the strategy."""
        if self.strategy == 'holdout':
            # Independent random partitions: the test sets may overlap.
            for _ in range(self.cv):
                yield train_test_split(self.dataset, self.split)
            return

        n = self.dataset.X.shape[0]
        if not 2 <= self.cv <= n:
            raise ValueError(
                f"cv={self.cv} is not usable with {n} samples: "
                "it must be at least 2 and at most the number of samples.")
        # Shuffle once, then cut into cv contiguous blocks. array_split copes
        # with n not dividing evenly by cv: the first n % cv folds get one
        # extra sample, so no sample is dropped and none is duplicated.
        indices = np.arange(n)
        np.random.default_rng(self.random_state).shuffle(indices)
        folds = np.array_split(indices, self.cv)
        for i in range(self.cv):
            test_idx = folds[i]
            # train on everything except fold i -- this is what makes the
            # folds disjoint, and what plain repeated splitting does not do
            train_idx = np.concatenate(
                [folds[j] for j in range(self.cv) if j != i])
            yield self._subset(train_idx), self._subset(test_idx)

    def run(self):
        train_scores = []
        test_scores = []
        ds = []
        for train, test in self._splits():
            ds.append((train, test))
            # A fresh copy per round: refitting the caller's instance would
            # leave it holding the last fold's parameters, and any state kept
            # from the previous round could leak into this one.
            model = copy.deepcopy(self.model)
            model.fit(train)
            if not self.score:
                train_scores.append(model.cost())
                test_scores.append(model.cost(test.X, test.y))
            else:
                # predict_all honours the model's predict convention; the old
                # apply_along_axis call fed single samples to every model and
                # so broke on batch predictors such as NaiveBayes and LDA.
                train_scores.append(self.score(train.y, predict_all(model, train.X)))
                test_scores.append(self.score(test.y, predict_all(model, test.X)))
        self.train_scores = train_scores
        self.test_scores = test_scores
        self.ds = ds
        return train_scores, test_scores

    def toDataframe(self):
        import pandas as pd
        assert self.train_scores and self.test_scores, "Need to run first"
        return pd.DataFrame({'Train Scores': self.train_scores,
                             'Test Scores': self.test_scores})


class GridSearchCV:

    def __init__(self, model, dataset, parameters, **kwargs):
        self.model = model
        self.dataset = dataset
        hasparam = [hasattr(self.model, param) for param in parameters]
        if np.all(hasparam):
            self.parameters = parameters
        else:
            index = hasparam.index(False)
            keys = list(parameters.keys())
            raise ValueError(f" Wrong parameters: {keys[index]}")
        self.kwargs = kwargs
        self.results = None

    def run(self):
        self.results = []
        attrs = list(self.parameters.keys())
        values = list(self.parameters.values())
        # itertools.product enumerates the full grid: every combination of one
        # value per hyper-parameter.
        for conf in itertools.product(*values):
            # Configure a FRESH copy of the model for each grid point. Mutating
            # and refitting one shared instance let state from the previous
            # configuration (fitted weights, is_fitted, the cached dataset)
            # carry over, which made a grid point's score depend on the order
            # the grid happened to be traversed in.
            model = copy.deepcopy(self.model)
            for attr, value in zip(attrs, conf):
                setattr(model, attr, value)
            scores = CrossValidationScore(model, self.dataset, **self.kwargs).run()
            self.results.append((conf, scores))
        return self.results

    def toDataframe(self):
        import pandas as pd
        assert self.results, "The grid search needs to be ran."
        data = dict()
        for i, k in enumerate(self.parameters.keys()):
            v = []
            for r in self.results:
                v.append(r[0][i])
            data[k] = v
        for i in range(len(self.results[0][1][0])):
            v = []
            t = []
            for r in self.results:
                v.append(r[1][0][i])
                t.append(r[1][1][i])
            data[f'CV_{i + 1} train'] = v
            data[f'CV_{i + 1} test'] = t

        return pd.DataFrame(data)
