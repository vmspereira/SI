from .transformer import Transformer
import numpy as np


class LabelEncoder(Transformer):
    """Encodes categorical labels as consecutive integers (0, 1, 2, ...).

    Example: ['cat', 'dog', 'cat', 'bird'] -> [1, 2, 1, 0].

    Use this when a model expects numeric targets and the classes have no
    meaningful order, or simply as the first step before one-hot encoding.
    Caveat: the integers imply an ordering (2 > 1 > 0) that the categories may
    not actually have, so for nominal INPUT features a one-hot encoding is
    usually safer. Integer labels are, however, fine as classification TARGETS.
    """

    def __init__(self) -> None:
        super().__init__()

    def fit(self, dataset):
        # Learn the set of distinct classes. np.unique also returns them
        # sorted, so the integer codes are assigned in a stable, repeatable
        # order.
        y = dataset.y
        self.classes = np.unique(y)
        return self

    def transform(self, dataset, inline: bool = False):
        # Build a lookup {class_value -> integer code} from the fitted classes.
        _map = {val: i for i, val in enumerate(self.classes)}
        # Replace every label with its integer code.
        _y = np.array([_map[x] for x in dataset.y])
        if inline:
            dataset.y = _y
            return dataset
        else:
            from .dataset import Dataset
            from copy import copy
            return Dataset(copy(dataset.X),
                           _y,
                           copy(dataset._xnames),
                           copy(dataset._yname)
                           )


class OneHotEncoder(Transformer):
    """Encodes integer labels as one-hot vectors.

    Example with 3 classes: label 0 -> [1, 0, 0], 1 -> [0, 1, 0],
    2 -> [0, 0, 1].

    Unlike LabelEncoder, one-hot encoding introduces NO artificial ordering
    between categories -- every class is equidistant from every other. This is
    the right choice for nominal categories fed as model inputs, and is the
    expected target format for many multi-class classifiers (e.g. softmax
    outputs). The trade-off is a wider representation: one column per class.
    Note this encoder expects the labels to already be integers (e.g. the
    output of LabelEncoder).
    """

    def transform(self, dataset, inline: bool = False):
        # Number of distinct classes = highest label index + 1 (labels are
        # assumed to be 0-based integers).
        n_values = np.max(dataset.y) + 1
        # np.eye(n) is the n x n identity matrix; its row k is exactly the
        # one-hot vector for class k. Indexing it with the label array picks
        # the right row for each sample, yielding shape (n_samples, n_values).
        _y = np.eye(n_values)[dataset.y]
        if inline:
            dataset.y = _y
            return dataset
        else:
            from .dataset import Dataset
            from copy import copy
            return Dataset(copy(dataset.X),
                           _y,
                           copy(dataset._xnames),
                           copy(dataset._yname)
                           )
