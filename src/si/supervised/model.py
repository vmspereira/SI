# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Created By  : Vítor Pereira
# Created Date: 01-09-2021
# version ='0.0.1'
# ---------------------------------------------------------------------------
"""Defines an interface for supervised learning models"""
# ---------------------------------------------------------------------------

from abc import ABC, abstractmethod


class Model(ABC):

    def __init__(self):
        """ Abstract class defining an interface for
        supervised learning models.

        A model needs to implement a `fit`, a `predict` and a `cost` method.
        """
        self.is_fitted = False

    @abstractmethod
    def fit(self, dataset, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def predict(self, x):
        raise NotImplementedError

    @abstractmethod
    def cost(self, *args, **kwarg):
        raise NotImplementedError
