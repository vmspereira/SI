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
    """Abstract base class defining the fit / transform contract.

    Every data-preprocessing object in the library (StandardScaler,
    VarianceThreshold, SelectKBest, the encoders, ...) is a Transformer and
    therefore shares the SAME three-method API. Learning this contract once
    means you know how to use all of them:

      * fit(dataset)            -> LEARN any parameters needed from the data
                                   (e.g. the mean/std for a scaler, the feature
                                   scores for SelectKBest). Returns self so
                                   calls can be chained.
      * transform(dataset)      -> APPLY the learned transformation, producing
                                   the modified data. Must be implemented by
                                   every subclass (it is abstract).
      * fit_transform(dataset)  -> convenience: fit then transform in one go.

    Splitting fit from transform is what lets us learn parameters on the
    TRAINING set and reuse them unchanged on validation/test data, avoiding
    information leaking from test into train.
    """

    def fit(self, dataset):
        """Learns the transformer parameters (if any).

        Default no-op implementation: transformers that need no learned state
        (e.g. a stateless encoder) can simply inherit this.

        :param dataset: A dataset to learn from.
        """
        return self

    @abstractmethod
    def transform(self, dataset, inline: bool = False):
        """Transforms a dataset.

        Abstract: each subclass MUST provide its own transformation logic.

        :param dataset: A dataset to transform.
        :param inline: If the transformation is to be applied inline to the input dataset\
            or if a new transformed dataset is to be generated.
        """
        raise NotImplementedError

    def fit_transform(self, dataset, inline=False):
        # Learn the parameters, then immediately apply them. This is the most
        # common way to use a transformer on the training data.
        self.fit(dataset)
        return self.transform(dataset, inline=inline)
