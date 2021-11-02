import numpy as np
from copy import copy


class StandardScaler:
    """
    Standardize features by centering the mean to 0 and unit variance.
    The standard score of an instance is calculated by:
        z = (x - u) / s
    where u is the mean of the training data and s is the standard deviation.
    Standardizing data is often necessary before training many machine
    learning models to avoid problems like exploding/vanishing gradients and
    feature dominance.
    Attributes
    ----------
    _mean : numpy array of shape (n_features, )
        The mean of each feature in the training set.
    _var : numpy array of shape (n_features, )
        The variance of each feature in the training set.
    """

    def fit(self, dataset):
        """
        Calculate and store the mean and variance of each feature in the
        training set.
        Parameters
        ----------
        dataset : A Dataset object to be standardized
        """
        X = dataset.X
        self._mean = np.mean(X, axis=0)
        self._var = np.var(X, axis=0)

    def transform(self, dataset, inline=False):
        """
        Standardize data by subtracting out the mean and dividing by
        standard deviation calculated during fitting.
        Parameters
        ----------
        dataset : A Dataset object to be standardized
        Returns
        -------
        A Dataset object with standardized data.
        """
        X = dataset.X
        Z = (X - self._mean) / np.sqrt(self._var)

        if inline:
            dataset.X = Z
            return dataset
        else:
            from . import Dataset
            return Dataset(Z,
                           copy(dataset.y),
                           copy(dataset._xnames),
                           copy(dataset._yname))

    def fit_transform(self, dataset, inline=False):
        """
        Calculate and store the mean and variance of each feature and
        standardize the data.
        Parameters
        ----------
        dataset : A Dataset object to be standardized
        Returns
        -------
        A Dataset object to with standardized data.
        """
        self.fit(dataset)
        return self.transform(dataset, inline=inline)

    def inverse_transform(self, dataset, inline=False):
        """
        Transform data back into orginal state by multiplying by standard
        deviation and adding the mean back in.
        Inverse standard scaler:
            x = z * s + u
        where s is the standard deviation, and u is the mean.
        Parameters
        ----------
        dataset : A standardized Dataset object
        Returns
        -------
        Dataset object
        """
        X = dataset.X * np.sqrt(self._var) + self._mean
        if inline:
            dataset.X = X
            return dataset
        else:
            from . import Dataset
            return Dataset(X,
                           copy(dataset.y),
                           copy(dataset._xnames),
                           copy(dataset._yname))
