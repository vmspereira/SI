# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Created By  : Vítor Pereira
# Created Date: 01-09-2021
# version ='0.0.1'
# ---------------------------------------------------------------------------
"""Supervised learning module"""
# ---------------------------------------------------------------------------
from .knn import KNN
from .linreg import LinearRegression, LinearRegressionReg
from .logreg import LogisticRegression, LogisticRegressionReg
from .dt import DecisionTree
from .ensemble import Ensemble, majority, average
from .nn import *
