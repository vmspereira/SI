# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Created By  : Vítor Pereira
# Created Date: 01-09-2021
# version ='0.0.1'
# ---------------------------------------------------------------------------
"""Utility module"""
# ---------------------------------------------------------------------------
import numpy as np

# Y is reserved to idenfify dependent variables
ALPHA = 'ABCDEFGHIJKLMNOPQRSTUVWXZ'


def label_gen(n: int):
    """
    Generates a list of n distinct labels similar to the ones
    on spreadsheets.
    Uses python yield for ease of computation.

    :param (int) n: Number of labels
    :returns: A list of labels

    Note: the import used to sit above this text, which silently demoted the
          docstring to a no-op string expression -- `label_gen.__doc__` was
          None and `help(label_gen)` showed nothing.
    """
    import itertools

    def _iter_all_strings():
        size = 1
        while True:
            for s in itertools.product(ALPHA, repeat=size):
                yield "".join(s)
            size += 1

    generator = _iter_all_strings()

    def gen():
        for s in generator:
            return s

    return [gen() for _ in range(n)]


def l1_distance(x, y):
    """
    Computes the manhatan distance of a point (x) to a set of
    points y.
    x.shape=(n,) and y.shape=(m,n)
    """
    import numpy as np
    dist = (np.absolute(x - y)).sum(axis=1)
    return dist


def l2_distance(x, y):
    """
    Computes the euclidean distance of a point (x) to a set of
    points y.
    x.shape=(n,) and y.shape=(m,n)

    :param x: a numpy.array
    :param y: a numpy.array
    :returns: a numpy.array of distances
    """
    dist = ((x - y) ** 2).sum(axis=1)
    return dist


def train_test_split(dataset, split: float = 0.8):
    """
    Splits randomly a dataset into a train and test set.

    :param dataset: The dataset to be splited.
    :param split: The percentage of samples to be used for training.
    """
    from ..data import Dataset
    n = dataset.X.shape[0]
    m = int(split * n)
    arr = np.arange(n)
    np.random.shuffle(arr)
    train_mask = arr[:m]
    test_mask = arr[m:]

    train = Dataset(dataset.X[train_mask], dataset.y[train_mask], dataset._xnames, dataset._yname)
    test = Dataset(dataset.X[test_mask], dataset.y[test_mask], dataset._xnames, dataset._yname)
    return train, test


def predict_all(model, X):
    """Predictions for every row of `X`, whichever convention `model` uses.

    Models in this library disagree about what `predict` takes: some want a
    single 1-D sample, others a whole 2-D batch (see `Model.predicts_batch`).
    Anything that scores an arbitrary model has to cope with both, and the
    shape of the data cannot settle it -- so read the model's declaration.

    :param model: a fitted model.
    :param X: numpy.array of shape (n_samples, n_features).
    :returns: numpy.array with one prediction per row of X.
    """
    if getattr(model, 'predicts_batch', False):
        return np.asarray(model.predict(X))
    return np.asarray([model.predict(row) for row in X])


def get_random_subsets(X, y, n_subsets, replacements=True):
    """ Return random subsets (with replacements) of the data """
    n_samples = np.shape(X)[0]
    # Concatenate x and y and do a random shuffle
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    subsets = []

    # Uses 50% of training samples without replacements
    subsample_size = int(n_samples // 2)
    if replacements:
        subsample_size = n_samples

    for _ in range(n_subsets):
        idx = np.random.choice(
            indices,
            size=np.shape(range(subsample_size)),
            replace=replacements)
        X_ = X[idx]
        y_ = y[idx]
        subsets.append((X_, y_))
    return subsets


def add_intercept(X):
    """
    Adds a column of "1" in front of a matrix (the intercept/bias term):

    | a b |  to  |1 a b |
    | c d |      |1 c d |
    :param X: numpy.array
    :returns: numpy.array
    """
    return np.hstack((np.ones((X.shape[0], 1)), X))


def sigmoid(z):
    """
    Sigmoid function
    :param z: a numpy.array
    :returns: a numpy array
    """
    return 1 / (1 + np.exp(-z))


def to_categorical(y, num_classes=None, dtype='float32'):
    """One-hot encodes a vector of integer class labels.

    Each label becomes a row of zeros with a single 1 in the column of its
    class, so label 2 out of 4 classes becomes [0, 0, 1, 0]. This is the target
    format a softmax output layer expects: the network produces a probability
    per class, and the one-hot row says which of those should be 1.

        y = [0, 2, 1]  ->  [[1, 0, 0],
                            [0, 0, 1],
                            [0, 1, 0]]

    Args:
        y: array-like of integer labels, shape (n_samples,) or
            (n_samples, 1). A trailing axis of size 1 is squeezed out.
        num_classes (int, optional): number of columns to produce. Defaults to
            max(y) + 1, which is only correct when every class appears in `y` --
            pass it explicitly when encoding a subset such as a single batch.
        dtype (str, optional): dtype of the result. Defaults to 'float32'.

    Returns:
        numpy.array of shape (n_samples, num_classes).
    """
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


def minibatch(X, y=None, batchsize=256, shuffle=True):
    if y is not None:
        assert X.shape[0] == y.shape[0]
    indices = np.arange(X.shape[0])
    batch_size = batchsize if batchsize < X.shape[0] else X.shape[0]
    if shuffle:
        np.random.shuffle(indices)
    for start_idx in range(0, X.shape[0] - batch_size + 1, batch_size):
        excerpt = indices[start_idx:start_idx + batch_size]
        if y is not None:
            yield X[excerpt], y[excerpt]
        else:
            yield X[excerpt]
