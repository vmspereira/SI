# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Created By  : Vítor Pereira
# Created Date: 01-09-2021
# version ='0.0.1'
# ---------------------------------------------------------------------------
"""Dataset module"""
# ---------------------------------------------------------------------------
import numpy as np
from ..util.util import label_gen


class Dataset:

    def __init__(self, X=None, y=None,
                 xnames: list = None,
                 yname: str = None):
        """ Tabular Dataset.

        This is the core data container passed to every model and transformer
        in the library. It bundles together:
          * X : the feature matrix, shape (n_samples, n_features).
          * y : the target/label vector, shape (n_samples,), or None for
                unlabeled (unsupervised) data.
          * xnames : the name of each feature column.
          * yname  : the name of the target column.
        Keeping names alongside the arrays lets transformers (e.g. feature
        selection) track WHICH features survive a transformation.
        """
        if X is None:
            raise Exception("Trying to instantiate a Dataset without any data")
        self.X = X
        self.y = y
        # If no feature names are supplied, auto-generate placeholders (one per
        # column of X); default the target name to 'y'.
        self._xnames = xnames if xnames else label_gen(X.shape[1])
        self._yname = yname if yname else 'y'

    @classmethod
    def from_data(cls, filename, sep=",", labeled=True):
        """Creates a Dataset from a data file.

        :param filename: The filename
        :type filename: str
        :param sep: attributes separator, defaults to ","
        :type sep: str, optional
        :return: A Dataset object
        :rtype: Dataset
        """
        data = np.genfromtxt(filename, delimiter=sep)
        if labeled:
            # Convention: when the data is labeled, the LAST column is the
            # target y and all preceding columns are the features X.
            X = data[:, 0:-1]
            y = data[:, -1]
        else:
            X = data
            y = None
        return cls(X, y)

    @classmethod
    def from_dataframe(cls, df, ylabel=None):
        """Creates a Dataset from a pandas dataframe.

        :param df: the source dataframe.
        :type df: pandas.DataFrame
        :param ylabel: name of the column to use as the label/target, defaults to None
        :type ylabel: str, optional
        :return: a Dataset object
        :rtype: Dataset
        """

        if ylabel and ylabel in df.columns:
            X = df.loc[:, df.columns != ylabel].to_numpy()
            y = df.loc[:, ylabel].to_numpy()
            xnames = list(df.columns)
            xnames.remove(ylabel)
            yname = ylabel
        else:
            X = df.to_numpy()
            y = None
            xnames = list(df.columns)
            yname = None
        return cls(X, y, xnames, yname)

    def __len__(self):
        """Returns the number of data points."""
        return self.X.shape[0]

    def hasLabel(self):
        """Returns True if the dataset constains labels (a dependent variable)"""
        return self.y is not None

    def getNumFeatures(self):
        """Returns the number of features"""
        return self.X.shape[1]

    def getNumClasses(self):
        """Returns the number of label classes or 0 if the dataset has no dependent variable."""
        return len(np.unique(self.y)) if self.hasLabel() else 0

    def writeDataset(self, filename, sep=","):
        """Saves the dataset to a file

        :param filename: The output file path
        :type filename: str
        :param sep: The fields separator, defaults to ","
        :type sep: str, optional
        """
        if self.y is not None:
            fullds = np.hstack((self.X, self.y.reshape(len(self.y), 1)))
        else:
            fullds = self.X
        np.savetxt(filename, fullds, delimiter=sep)

    def toDataframe(self):
        """ Converts the dataset into a pandas DataFrame"""
        import pandas as pd
        if self.y is not None:
            fullds = np.hstack((self.X, self.y.reshape(len(self.y), 1)))
            columns = self._xnames[:] + [self._yname]
        else:
            fullds = self.X.copy()
            columns = self._xnames[:]
        return pd.DataFrame(fullds, columns=columns)

    def __repr_html__(self) -> str:
        return self.toDataframe().to_html()

    def getXy(self):
        # Convenience accessor returning the (features, target) pair, the form
        # most models consume in their `fit` method.
        return self.X, self.y


def summary(dataset, format='df'):
    """ Returns the statistics of a dataset(mean, std, max, min)

    Computes per-column descriptive statistics so students can quickly inspect
    the scale and spread of each feature (and the target) -- handy for spotting
    whether standardization or outlier removal is needed before modelling.

    :param dataset: A Dataset object
    :type dataset: si.data.Dataset
    :param format: Output format ('df':DataFrame, 'dict':dictionary ), defaults to 'df'
    :type format: str, optional
    """
    if dataset.hasLabel():
        # Glue the target on as an extra column so it is summarized too.
        # reshape turns y from (n,) into (n, 1) so hstack can append it.
        fullds = np.hstack((dataset.X, dataset.y.reshape(len(dataset.y), 1)))
        columns = dataset._xnames[:] + [dataset._yname]
    else:
        fullds = dataset.X
        columns = dataset._xnames[:]
    stats = {}
    # Walk every column and record its mean, variance, min and max.
    for i in range(fullds.shape[1]):
        try:
            _means = np.mean(fullds[:, i], axis=0)
            _vars = np.var(fullds[:, i], axis=0)
            _maxs = np.max(fullds[:, i], axis=0)
            _mins = np.min(fullds[:, i], axis=0)
        except Exception:
            _means = _vars = _maxs = _mins = np.nan
        stat = {'mean': _means,
                'var': _vars,
                'min': _mins,
                'max': _maxs
                }
        stats[columns[i]] = stat
    if format == 'df':
        import pandas as pd
        df = pd.DataFrame(stats)
        return df
    else:
        return stats
