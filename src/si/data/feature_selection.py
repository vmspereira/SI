# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Created By  : Vítor Pereira
# Created Date: 01-09-2021
# version ='0.0.1'
# ---------------------------------------------------------------------------
"""Feature Selection module"""
# ---------------------------------------------------------------------------
from .transformer import Transformer
import numpy as np
from scipy import stats
from copy import copy


class VarianceThreshold(Transformer):

    def __init__(self, threshold=0):
        """
        The variance threshold is a simple baseline approach to feature selection.
        It removes all features whose variance doesn't meet some threshold. By default,
        it removes all zero-variance features, i.e., features that have the same value in all samples.

        Intuition: a feature that barely varies carries almost no information
        to distinguish samples, so it is unlikely to help any model and can be
        dropped. Note this is UNSUPERVISED -- it looks only at X, never at the
        labels y -- so it cannot tell whether a high-variance feature is
        actually predictive, only that it changes across samples.

        :param threshold: The non negative threshold value, defaults to 0.
        :type threshold: int, optional
        """
        # A negative threshold used to only warn and then be kept, so
        # `var > threshold` was true for everything and the transformer became a
        # silent no-op -- it even retained the zero-variance features it exists
        # to remove. Warning without acting is the worst of both, so reject it.
        if threshold < 0:
            raise ValueError(
                f"threshold must be non-negative (it is compared against a "
                f"variance, which cannot be negative); got {threshold}."
            )
        self.threshold = threshold

    def fit(self, dataset):
        # Compute the variance of every feature (one value per column).
        X = dataset.X
        self._var = np.var(X, axis=0)
        return self

    def transform(self, dataset, inline=False):
        assert hasattr(self, '_var'), \
            'VarianceThreshold must be fit before transforming'
        X = dataset.X
        # Boolean mask: True for features we keep (variance above threshold).
        cond = self._var > self.threshold
        # Positions of the kept features...
        idxs = [i for i in range(len(cond)) if cond[i]]
        # ...used to slice the surviving columns out of X and keep their names.
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


def f_classif(dataset):
    """Scoring function for classifications.

    Compute the ANOVA F-value for the provided sample.

    We want to identify which groups have means
    significantly different.

    The null hypotesis, H0, it that the means
    is the same for all groups, ie, the factors
    or features do not significantly affect the labels.

    The ANOVA F-test idea
    ---------------------
    For each feature we split the values by class label and compare:
        F = (variance BETWEEN the group means) / (variance WITHIN the groups)
    If a feature is useful for classification, the class means will be far
    apart (large between-group variance) relative to the spread inside each
    class (small within-group variance), giving a LARGE F. If the feature is
    irrelevant, the group means look the same and F is near 1. A large F
    therefore translates into a small p-value (strong evidence against H0,
    i.e. the feature does discriminate between classes).

    :param dataset: A labeled dataset
    :type dataset: Dataset
    :return: F scores and p-values
    :rtype: a tupple of np.arrays
    """
    X = dataset.X
    y = dataset.y
    # Groups the data entries by lable class: args[c] holds the rows of X
    # whose label equals class c (one sub-array per class).
    args = [X[y == a] for a in np.unique(y)]
    # Computes the F-statistics and p values. f_oneway runs a one-way ANOVA
    # per feature, returning one F and one p-value for each column.
    F, p = stats.f_oneway(*args)
    return F, p


def f_regress(dataset):
    """Scoring function for regressions

    F-test for regression

    The null hypotesis, in this case,
    is that all coefficientes are zero, in other words,
    the model does not have predictive capabilities.

    Here the score for each feature is derived from its linear correlation
    with the continuous target y: a feature that correlates strongly with y is
    a good predictor. The correlation is converted into an F-statistic, so the
    output is comparable in spirit to f_classif (large F = informative
    feature, small p-value).

    :param dataset: A labeled dataset
    :type dataset: Dataset
    :return: F scores and p-values
    :rtype: a tupple of np.arrays
    """
    X = dataset.X
    y = dataset.y
    # Pearson correlation coefficient r between each feature column and y.
    correlation_coefficient = np.array([stats.pearsonr(X[:, i], y)[0] for i in range(X.shape[1])])
    # Degrees of freedom for a simple linear regression: n samples minus the
    # 2 estimated parameters (slope and intercept).
    deg_of_freedom = y.size - 2
    corr_coef_squared = correlation_coefficient ** 2  # r^2, the explained variance fraction
    # Standard formula turning r^2 into an F-statistic. The ratio
    # r^2 / (1 - r^2) is "explained over unexplained" variance; scaling by the
    # degrees of freedom makes it follow an F distribution.
    #
    # A perfectly correlated feature has r^2 == 1, leaving no unexplained
    # variance and dividing by zero -- which returned inf. inf is not wrong in
    # spirit (the evidence is unbounded) but it propagates into any later
    # arithmetic, so it is capped at the largest finite float instead. Ranking is
    # unaffected: such a feature still sorts above every other.
    unexplained = 1 - corr_coef_squared
    with np.errstate(divide='ignore', invalid='ignore'):
        F = np.where(unexplained > 0,
                     corr_coef_squared / np.where(unexplained > 0, unexplained, 1.0)
                     * deg_of_freedom,
                     np.finfo(float).max)
    # Convert each F into a p-value via the survival function (1 - CDF) of the
    # F distribution with (1, deg_of_freedom) degrees of freedom.
    p = stats.f.sf(F, 1, deg_of_freedom)
    return F, p


class SelectKBest(Transformer):

    def __init__(self, k: int, score_func=f_classif):
        """The SelectKBest method selects the features according to the k highest scores
        computed using a scoring function.

        :param k: Number of feature with best score to be selected
        :type k: int
        :param score_func: The scoring function, defaults to f_classif
        :type score_func: callable, optional

        -------------------------------------------------------------------------
        In this implementation we will consider the two F-statistics functions,
        one for regression (f_regress) and the other for classification tasks (f_classif).

        The p and F values have an inverse relationship, the greater
        the F value the lesser the p.
        Larger values of F correspond to a rejection with probability
        (1-p) of the null hypothesis, meaning that the corresponding
        features has an effect on the predictions.
        """
        self.k = k
        self.score_func = score_func

    def fit(self, dataset):
        # Score every feature with the chosen function (ANOVA for
        # classification, correlation-based F for regression).
        self.F, self.p = self.score_func(dataset)
        n_features = dataset.X.shape[1]
        if not 1 <= self.k <= n_features:
            raise ValueError(
                f"k must be between 1 and the number of features ({n_features}); "
                f"got {self.k}. Note k=0 previously kept EVERY feature, because "
                "argsort()[-0:] is the whole array rather than an empty slice."
            )
        return self

    def transform(self, dataset, inline=False):
        assert hasattr(self, 'F'), 'SelectKBest must be fit before transforming'
        # identify the k features with higher F values.
        #
        # A constant (zero-variance) feature makes the ANOVA degenerate and
        # scores nan. np.argsort places nan LAST, i.e. exactly where the highest
        # scores live, so `[-k:]` used to pick the uninformative columns FIRST:
        # given F = [13.96, 0.11, 1.00, nan], k=1 selected the nan column and
        # discarded the genuinely predictive one. Replacing nan with -inf sends
        # those features to the bottom of the ranking, where they belong.
        scores = np.where(np.isnan(self.F), -np.inf, self.F)
        # argsort returns indices that sort scores ascending; [-self.k:] takes
        # the last k, i.e. the k highest-scoring (most informative) features.
        idxs = scores.argsort()[-self.k:]
        # Re-sort the kept indices so the selected columns stay in their
        # original feature order.
        idxs.sort()
        # Slice out just those k columns and keep their matching names.
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
