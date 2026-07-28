from .model import Model
from si.data.transformer import Transformer
import numpy as np

class LDA(Model,Transformer):
    
    def __init__(self):
        super().__init__()
        self.w = None
        
    def fit(self, dataset, **kwargs):
        self.dataset = dataset
        X, y = dataset.getXy()
        # Binary Fisher LDA: separate the two classes
        X1 = X[y == 0]
        X2 = X[y == 1]

        # Within-class scatter as the sum of the per-class covariance matrices
        # (rowvar=False -> variables are the columns/features).
        cov1 = np.cov(X1, rowvar=False)
        cov2 = np.cov(X2, rowvar=False)
        cov_tot = np.atleast_2d(cov1 + cov2)

        # Class means
        mean1 = X1.mean(0)
        mean2 = X2.mean(0)
        mean_diff = np.atleast_1d(mean1 - mean2)

        # Discriminant direction: w = Sw^-1 (mean1 - mean2)
        self.w = np.linalg.pinv(cov_tot).dot(mean_diff)
        # Decision boundary at the midpoint of the projected class means. w points
        # from class 1 toward class 0, so w.mean1 >= threshold >= w.mean2.
        self.threshold = self.w.dot((mean1 + mean2) / 2)
        return self

    def transform(self, dataset, inline: bool = False):
        X_transform = dataset.X.dot(self.w)
        if inline:
            dataset.X = X_transform
        return X_transform

    def predict(self, X):
        y_pred = []
        for sample in X:
            h = sample.dot(self.w)
            # projection above the midpoint -> class 0, below -> class 1
            y = int(h < self.threshold)
            y_pred.append(y)
        return y_pred
    
    def cost(self, *args, **kwarg):
        return super().cost(*args, **kwarg)