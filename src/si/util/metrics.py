# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Created By  : Vítor Pereira
# Created Date: 01-09-2021
# version ='0.0.1'
# ---------------------------------------------------------------------------
"""Metrics module"""
# ---------------------------------------------------------------------------

import numpy as np


def accuracy_score(y_true, y_pred):
    """
    Classification performance metric that computes the accuracy of y_true
    and y_pred.

    :param numpy.array y_true: array-like of shape (n_samples,) Ground truth correct labels.
    :param numpy.array y_pred: array-like of shape (n_samples,) Estimated target values.
    :returns: C (float) Accuracy score.
    """
    accuracy = (y_true, y_pred).sum() / len(y_true)
    return accuracy


def mse(y_true, y_pred):
    """
    Mean squared error regression loss function.
    Parameters

    :param numpy.array y_true: array-like of shape (n_samples,)
        Ground truth (correct) target values.
    :param numpy.array y_pred: array-like of shape (n_samples,)
        Estimated target values.
    :returns: loss (float) A non-negative floating point value (the best value is 0.0).
    """
    return np.mean(np.power(y_true-y_pred, 2))


def mse_prime(y_true, y_pred):
    return 2*(y_pred-y_true)/y_true.size


def cross_entropy(y_true, y_pred):
    m = y_pred.shape[0]
    return -(y_true * np.log(y_pred)).sum()/m


def cross_entropy_prime(y_true, y_pred):
    m = y_pred.shape[0]
    return (y_pred - y_true)/m


def r2_score(y_true, y_pred):
    """
    R^2 regression score function.
        R^2 = 1 - SS_res / SS_tot
    where SS_res is the residual sum of squares and SS_tot is the total
    sum of squares.

    :param numpy.array y_true : array-like of shape (n_samples,) Ground truth (correct) target values.
    :param numpy.array y_pred : array-like of shape (n_samples,) Estimated target values.
    :returns: score (float) R^2 score.
    """
    # Residual sum of squares.
    numerator = ((y_true - y_pred) ** 2).sum(axis=0)
    # Total sum of squares.
    denominator = ((y_true - np.average(y_true, axis=0)) ** 2).sum(axis=0)
    # R^2.
    score = 1 - numerator / denominator
    return score
