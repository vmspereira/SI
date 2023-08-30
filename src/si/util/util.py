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


def label_gen(n:int):
    import itertools
    """ 
    Generates a list of n distinct labels similar to the ones
    on spreadsheets.
    Uses python yield for ease of computation.

    :param (int) n: Number of labels
    :returns: A list of labels
    """
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

def train_test_split(dataset, split:float=0.8):
    """
    Splits randomly a dataset into a train and test set.

    :param dataset: The dataset to be splited.
    :param split: The percentage of samples to be used for training.
    """
    from ..data import Dataset
    n = dataset.X.shape[0]
    m = int(split*n)
    arr = np.arange(n)
    np.random.shuffle(arr)
    train_mask = arr[:m]
    test_mask = arr[m:]

    train = Dataset(dataset.X[train_mask], dataset.y[train_mask], dataset._xnames, dataset._yname)
    test = Dataset(dataset.X[test_mask], dataset.y[test_mask], dataset._xnames, dataset._yname)
    return train, test

def add_intersect(X):
    """ 
    Adds a vector of "1" in front of a matrix:

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
    """_summary_

    Args:
        y (_type_): _description_
        num_classes (_type_, optional): _description_. Defaults to None.
        dtype (str, optional): _description_. Defaults to 'float32'.

    Returns:
        _type_: _description_
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

def minibatch(X,y, batchsize=256, shuffle=True):
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
