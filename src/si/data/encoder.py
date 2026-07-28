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
        assert hasattr(self, 'classes'), \
            'LabelEncoder must be fit before transforming'
        # Build a lookup {class_value -> integer code} from the fitted classes.
        _map = {val: i for i, val in enumerate(self.classes)}
        # A label absent from the fitted classes -- typically a class that only
        # appears in the test split -- used to surface as a bare
        # "KeyError: 'z'", which says nothing about the cause.
        unseen = sorted({str(x) for x in dataset.y if x not in _map})
        if unseen:
            raise ValueError(
                f"Cannot encode label(s) {unseen}: not seen during fit. Known "
                f"classes are {[str(c) for c in self.classes]}. Fit on data "
                "covering every class (or on the full label set) before "
                "transforming."
            )
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

    def fit(self, dataset):
        """Records how many columns the encoding needs.

        Calling fit first is what makes the width consistent between splits: the
        number of classes is taken from the data seen HERE and reused by every
        later transform. Without it, transform falls back to the maximum label in
        whatever data it is handed, so a batch that happens to lack the top class
        produces a narrower encoding than the training data did.
        """
        self.n_values = int(np.max(dataset.y)) + 1
        return self

    def transform(self, dataset, inline: bool = False):
        y = np.asarray(dataset.y)

        # np.eye is indexed with the labels, and numpy reads a NEGATIVE index as
        # counting from the end -- so label -1 silently produced the one-hot
        # vector of the LAST class instead of raising. Non-integer labels give a
        # confusing TypeError from the indexing itself.
        if y.dtype.kind not in 'iu':
            if y.dtype.kind == 'f' and np.all(y == np.floor(y)):
                y = y.astype(int)
            else:
                raise ValueError(
                    "OneHotEncoder expects 0-based integer labels (e.g. the "
                    f"output of LabelEncoder); got dtype {y.dtype}."
                )
        if y.min() < 0:
            raise ValueError(
                f"OneHotEncoder expects 0-based non-negative labels; got {y.min()}. "
                "A negative label would be read as an index from the end and "
                "silently encoded as the wrong class."
            )

        # Number of distinct classes: the width learned in fit if there was one,
        # otherwise the highest label index + 1 in this data.
        n_values = getattr(self, 'n_values', None)
        if n_values is None:
            n_values = int(y.max()) + 1
        elif y.max() >= n_values:
            raise ValueError(
                f"Label {y.max()} does not fit the {n_values} columns learned "
                "during fit. Fit on data covering every class."
            )
        # np.eye(n) is the n x n identity matrix; its row k is exactly the
        # one-hot vector for class k. Indexing it with the label array picks
        # the right row for each sample, yielding shape (n_samples, n_values).
        _y = np.eye(n_values)[y]
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
