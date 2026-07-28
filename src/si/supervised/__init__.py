# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Created By  : Vítor Pereira
# Created Date: 01-09-2021
# version ='0.0.1'
# ---------------------------------------------------------------------------
"""Supervised learning module"""
# ---------------------------------------------------------------------------
from .knn import KNN
from .linreg import LinearRegression
from .logreg import LogisticRegression
from .dt import DecisionTree
from .rf import RandomForest
from .nb import NaiveBayes
from .lda import LDA
from .ensemble import Ensemble, majority, average
from .nn import *

# Note: SVM is intentionally not imported here because it hard-depends on
# cvxopt; import it explicitly via `from si.supervised.svm import SVM`.
