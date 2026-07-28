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
        X = np.asarray(X)
        # A 1-D X used to fail on X.shape[1] with "IndexError: tuple index out of
        # range", which does not hint at the shape being the problem.
        #
        # Only 1-D (and 0-D) is rejected. Higher-rank X is legitimate and used by
        # the library itself: the RNN takes a batch of SEQUENCES,
        # (n_samples, timesteps, features), and Conv2D takes a batch of IMAGES in
        # NHWC, (n_samples, rows, cols, channels). An earlier version of this
        # check demanded exactly 2-D and broke both -- scripts/eval6.ipynb, which
        # builds a (3000, 10, 20) sequence dataset, stopped running.
        if X.ndim < 2:
            raise ValueError(
                f"X must have at least 2 dimensions, the first being the samples;"
                f" got {X.ndim} dimension(s) with shape {X.shape}. A single "
                "feature needs an explicit column: X.reshape(-1, 1)."
            )
        if y is not None and len(y) != X.shape[0]:
            raise ValueError(
                f"X has {X.shape[0]} samples but y has {len(y)}; they must match."
            )
        # Feature names must line up with the columns, or the mismatch only
        # surfaces much later inside toDataframe/summary as an opaque pandas
        # shape error.
        if xnames is not None and len(xnames) != X.shape[1]:
            raise ValueError(
                f"xnames has {len(xnames)} name(s) but X has {X.shape[1]} "
                "column(s); they must match."
            )
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

        # A ylabel that is not a column used to fall through to the UNLABELED
        # branch, so a typo silently produced a dataset with no target and the
        # target column folded in among the features -- the worst possible
        # outcome for someone who explicitly asked for a label.
        if ylabel is not None and ylabel not in df.columns:
            raise ValueError(
                f"ylabel '{ylabel}' is not a column of the dataframe. "
                f"Available columns: {list(df.columns)}."
            )

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
            fullds = np.hstack((np.asarray(self.X, dtype=object),
                                np.asarray(self.y, dtype=object).reshape(-1, 1)))
        else:
            fullds = np.asarray(self.X, dtype=object)
        # '%s' rather than the default '%.18e': hstacking a float X with a string
        # y gives a text array, and the numeric format specifier then raised
        # "TypeError: Mismatch between array dtype ('<U32') and format specifier".
        # '%s' writes both kinds faithfully.
        np.savetxt(filename, fullds, delimiter=sep, fmt='%s')

    def toDataframe(self):
        """ Converts the dataset into a pandas DataFrame

        The target is attached as its own column rather than hstacked onto X.
        np.hstack forces one common dtype, so string labels used to upcast the
        numeric features to strings -- every column came back as text and
        `summary` then reported NaN for all of them.
        """
        import pandas as pd
        df = pd.DataFrame(self.X, columns=self._xnames[:])
        if self.y is not None:
            # assigning a column preserves both dtypes independently
            df[self._yname] = self.y
        return df

    def _repr_html_(self) -> str:
        """Rich display hook for Jupyter.

        Named with SINGLE leading/trailing underscores because that is what
        IPython looks up. As `__repr_html__` it was never called, so datasets
        rendered as a plain <object ...> in notebooks.
        """
        return self.toDataframe().to_html()

    def getXy(self):
        # Convenience accessor returning the (features, target) pair, the form
        # most models consume in their `fit` method.
        return self.X, self.y


def summary(dataset, output_format='df'):
    """ Returns the statistics of a dataset(mean, std, max, min)

    Computes per-column descriptive statistics so students can quickly inspect
    the scale and spread of each feature (and the target) -- handy for spotting
    whether standardization or outlier removal is needed before modelling.

    A non-numeric column (e.g. string labels) reports NaN for all four
    statistics, since a mean has no meaning there. Only that column is affected:
    the columns are gathered separately rather than hstacked into one array,
    because hstack forces a single dtype and a string target used to turn every
    numeric feature into text -- making the whole table NaN.

    :param dataset: A Dataset object
    :type dataset: si.data.Dataset
    :param output_format: Output format ('df':DataFrame, 'dict':dictionary ),
        defaults to 'df'. (Named `output_format` rather than `format` so it does
        not shadow the `format` builtin.)
    :type output_format: str, optional
    """
    # (name, column) pairs, each column keeping its own dtype
    columns = [(name, dataset.X[:, i])
               for i, name in enumerate(dataset._xnames)]
    if dataset.hasLabel():
        # Include the target so it is summarized too.
        columns.append((dataset._yname, np.asarray(dataset.y)))

    stats = {}
    # Walk every column and record its mean, variance, min and max.
    for name, column in columns:
        try:
            stat = {'mean': np.mean(column),
                    'var': np.var(column),
                    'min': np.min(column),
                    'max': np.max(column)}
        except (TypeError, ValueError):
            # non-numeric column: these statistics do not apply
            stat = {'mean': np.nan, 'var': np.nan,
                    'min': np.nan, 'max': np.nan}
        stats[name] = stat
    if output_format == 'df':
        import pandas as pd
        df = pd.DataFrame(stats)
        return df
    else:
        return stats
