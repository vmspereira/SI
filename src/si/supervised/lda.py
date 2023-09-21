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
         # Separate data by class
        X1 = X[y == 0]
        X2 = X[y == 1]
        
        # Calculate the covariance matrices of the two datasets
        X1_center = X1 - np.mean(X1, axis=0)
        X2_center = X2 - np.mean(X2, axis=0)
        cov1 = np.corrcoef(X1_center.T)
        cov2 = np.corrcoef(X2_center.T)
        cov_tot = cov1 + cov2

        # Calculate the mean of the two datasets
        mean1 = X1.mean(0)
        mean2 = X2.mean(0)
        mean_diff = np.atleast_1d(mean1 - mean2)

        # Determine the vector which when X is projected onto it best separates the
        # data by class. w = (mean1 - mean2) / (cov1 + cov2)
        self.w = np.linalg.pinv(cov_tot).dot(mean_diff)
    
    def transform(self, dataset, inline: bool = False):
        X_transform = dataset.X.dot(self.w)
        if inline: 
            dataset.X = X_transform 
        return X_transform

    
    def predict(self, X):
        y_pred = []
        for sample in X:
            h = sample.dot(self.w)
            y = 1 * (h < 0)
            y_pred.append(y)
        return y_pred
    
    def cost(self, *args, **kwarg):
        return super().cost(*args, **kwarg)