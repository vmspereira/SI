# ---------------------------------------------------------------------------
# Fisher's Linear Discriminant Analysis (binary case).
#
# Idea: find the single direction w onto which we can project the data so that,
# once projected to one dimension (h = w . x), the two classes are pushed as far
# apart as possible while each class stays tightly clustered. Formally Fisher
# maximises the ratio
#
#         (between-class separation)      (w . (mean1 - mean2))^2
#         --------------------------  =   -----------------------
#         (within-class scatter)              w^T Sw w
#
# whose solution is  w proportional to  Sw^-1 (mean1 - mean2), where Sw is the
# within-class scatter matrix (here the sum of the two class covariance
# matrices). After projecting, a 1-D threshold separates the classes.
#
# LDA doubles as a supervised dimensionality-reduction / feature extractor
# (hence it is also a Transformer): project onto w to get a 1-D feature that is
# maximally discriminative. Use it when classes are roughly Gaussian with
# similar covariance and you want a cheap linear baseline or projection.
# ---------------------------------------------------------------------------
from .model import Model
from si.data.transformer import Transformer
from si.util import accuracy_score
import numpy as np


class LDA(Model, Transformer):

    # `predict` takes a 2-D X (n_samples, n_features) and returns one
    # prediction per row -- see the note on Model.predicts_batch.
    predicts_batch = True

    def __init__(self):
        super().__init__()
        # w: the learned discriminant direction (set during fit).
        self.w = None
        # classes: the two labels seen during fit, in sorted order. Kept so
        # predict can return an actual label instead of a 0/1 position.
        self.classes = None

    def fit(self, dataset, **kwargs):
        """Learn the Fisher discriminant direction w and the decision threshold.

        Steps: split the two classes, build the within-class scatter Sw as the
        sum of the per-class covariances, then set w = Sw^-1 (mean1 - mean2) and
        place the threshold at the midpoint of the projected class means.
        """
        self.dataset = dataset
        X, y = dataset.getXy()
        # Binary Fisher LDA: separate the two classes.
        #
        # The two groups are selected by the labels actually present, not by
        # hard-coded 0 and 1. Comparing against literals meant that for any
        # other label pair -- {1, 2}, {5, 9}, {'cat', 'dog'} -- one of the
        # slices came back EMPTY, its mean was NaN, and the NaN propagated into
        # the scatter matrix until the pseudo-inverse died with
        # "LinAlgError: SVD did not converge".
        self.classes = np.unique(y)
        if len(self.classes) != 2:
            raise ValueError(
                "LDA implements the BINARY Fisher discriminant and needs "
                f"exactly 2 classes; got {len(self.classes)}: "
                f"{self.classes.tolist()}."
            )
        X1 = X[y == self.classes[0]]
        X2 = X[y == self.classes[1]]

        # Within-class scatter Sw as the sum of the per-class covariance matrices
        # (rowvar=False -> variables are the columns/features). Sw captures how
        # spread out each class is; minimising w^T Sw w keeps classes compact.
        # Shape: (n_features, n_features).
        cov1 = np.cov(X1, rowvar=False)
        cov2 = np.cov(X2, rowvar=False)
        cov_tot = np.atleast_2d(cov1 + cov2)

        # Class means (centroids), one value per feature.
        mean1 = X1.mean(0)
        mean2 = X2.mean(0)
        # Direction connecting the two centroids; this is what we want to
        # "amplify" relative to the within-class spread.
        mean_diff = np.atleast_1d(mean1 - mean2)

        # Discriminant direction: w = Sw^-1 (mean1 - mean2)
        # pinv (pseudo-inverse) is used instead of inv so this is robust even
        # when Sw is singular (e.g. fewer samples than features).
        self.w = np.linalg.pinv(cov_tot).dot(mean_diff)
        # Decision boundary at the midpoint of the projected class means. w points
        # from classes[1] toward classes[0], so w.mean1 >= threshold >= w.mean2.
        self.threshold = self.w.dot((mean1 + mean2) / 2)
        # fit() never set this, so is_fitted stayed False for the lifetime of the
        # model. Nothing noticed because neither predict nor cost checked it.
        self.is_fitted = True
        return self

    def transform(self, dataset, inline: bool = False):
        # Dimensionality reduction: project every sample onto w, collapsing the
        # n features to a single, maximally discriminative coordinate (X @ w).
        X_transform = dataset.X.dot(self.w)
        if inline:
            dataset.X = X_transform
        return X_transform

    def predict(self, X):
        """Classify each row of X by which side of the threshold it projects to.

        Returns the class LABELS, not 0/1 positions: w points from classes[1]
        towards classes[0], so a projection at or above the threshold belongs to
        classes[0] and one below it to classes[1].
        """
        assert self.is_fitted, 'Model must be fit before predicting'
        y_pred = []
        for sample in X:
            h = sample.dot(self.w)
            # position 0 above the midpoint, position 1 below it, then map the
            # position back to the label it stands for
            position = int(h < self.threshold)
            y_pred.append(self.classes[position])
        return np.asarray(y_pred)

    def cost(self, X=None, y=None):
        """Training-set accuracy (or the accuracy on supplied X, y).

        This used to be `return super().cost(...)`, which reaches the abstract
        method on Model and therefore raised NotImplementedError on every call.
        The override existed only to let the class be instantiated.
        """
        assert self.is_fitted, 'Model must be fit before predicting'
        X = X if X is not None else self.dataset.X
        y = y if y is not None else self.dataset.y
        # LDA is a batch predictor, so predict is called on the whole matrix.
        return accuracy_score(y, self.predict(X))
