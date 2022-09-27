# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Created By  : Vítor Pereira
# Created Date: 01-09-2021
# version ='0.0.1'
# ---------------------------------------------------------------------------
"""Feature Selection module"""
# ---------------------------------------------------------------------------
from abc import ABC, abstractmethod

class Transformer(ABC):

    @abstractmethod
    def fit(self, dataset):
        """Learns the transformer parameters (if any).

        :param dataset: A dataset to learn from.
        """
        raise NotImplementedError

    @abstractmethod
    def transform(self, dataset, inline:bool=False):
        """Transforms a dataset.

        :param dataset: A dataset to transform.
        :param inline: If the tranformation is to be applyied inline to the input dataset\
            or if a new transformed dataset is to be generated.
        """
        raise NotImplementedError

    def fit_transform(self, dataset, inline=False):
        self.fit(dataset)
        return self.transform(dataset, inline=inline)
