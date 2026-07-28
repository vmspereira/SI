# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Created By  : Vítor Pereira
# Created Date: 01-09-2021
# version ='0.0.1'
# ---------------------------------------------------------------------------
"""Metrics module"""
# ---------------------------------------------------------------------------

import numpy as np
import pandas as pd


def confusion_matrix(true_y, predict_y, format='df'):
    """
    Computes a confusion matrix
    """
    cm = pd.crosstab(true_y, predict_y,
                     rownames=["True values"],
                     colnames=["Predicted values"])
    if format == 'df':
        return pd.DataFrame(cm)
    else:
        return cm


def accuracy_score(y_true, y_pred):
    """
    Classification performance metric that computes the accuracy of y_true
    and y_pred.

    :param numpy.array y_true: array-like of shape (n_samples,) Ground truth correct labels.
    :param numpy.array y_pred: array-like of shape (n_samples,) Estimated target values.
    :returns: (float) Accuracy score.
    """
    accuracy = (y_true == y_pred).sum() / len(y_true)
    return accuracy


def multiclass_accuracy(y_true, y_pred):
    p = np.argmax(y_pred, axis=1)
    t = np.argmax(y_true, axis=1)
    return accuracy_score(t, p)


def mae(y_true, y_pred):
    """
    Mean absolute error loss function.
    Parameters

    :param numpy.array y_true: array-like of shape (n_samples,)
        Ground truth (correct) target values.
    :param numpy.array y_pred: array-like of shape (n_samples,)
        Estimated target values.
    :returns: loss (float) A non-negative floating point value (the best value is 0.0).
    """
    return np.mean(np.abs(y_true - y_pred))


def mae_prime(y_true, y_pred):
    X = y_true - y_pred
    m = y_pred.shape[0]
    return np.where(X > 0, -1 / m, np.where(X < 0, 1 / m, 0))


def mse(y_true, y_pred):
    """
    Mean squared error regression loss function.
    Parameters

    :param numpy.array y_true: array-like of shape (n_samples,)
        Ground truth (correct) target values.
    :param numpy.array y_pred: array-like of shape (n_samples,)
        Estimated target values.
    :returns: loss (float) A non-negative floating point value (the best value is 0.0).

    Note: some implementations of the MSE consider additionaly a division by 2
          to obtain a `cleaner` derivative allowing to cancel the factor '2'
          (see mse_prime).
          Computationally, they are equivalent as both require a bit shift.
    """
    return np.mean(np.power(y_true - y_pred, 2))


def mse_prime(y_true, y_pred):
    """ The derivative of the MSE.

    :param numpy.array y_true: array-like of shape (n_samples,)
        Ground truth (correct) target values.
    :param numpy.array y_pred: array-like of shape (n_samples,)
        Estimated target values.
    :returns: the derivative of the MSE

    Note: To avoid the additional multiplication by -1 just swap
          the y_pred and y_true.
    """
    return 2 * (y_pred - y_true) / y_true.size


def rmse(y_true, y_pred):
    """Rooted MSE

    :param numpy.array y_true: array-like of shape (n_samples,)
        Ground truth (correct) target values.
    :param numpy.array y_pred: array-like of shape (n_samples,)
        Estimated target values.
    :returns: RMSE
    """
    return np.sqrt(mse(y_true, y_pred))


def rmse_prime(y_true, y_pred):
    """Derivative of RMSE

    :param numpy.array y_true: array-like of shape (n_samples,)
        Ground truth (correct) target values.
    :param numpy.array y_pred: array-like of shape (n_samples,)
        Estimated target values.
    :returns: the derivative of the RMSE
    """
    X = (y_pred - y_true)
    return np.where(X == 0, 0, X / (rmse(y_true, y_pred) * y_true.size))


def cross_entropy(y_true, y_pred, eps=1e-15):
    """Cross entropy

    :param numpy.array y_true: array-like of shape (n_samples,)
        Ground truth (correct) target values.
    :param numpy.array y_pred: array-like of shape (n_samples,)
        Estimated target values.
    :param float eps: clipping bound that keeps the logarithm finite.
    :returns: cross entropy score

    Note: log(0) is -inf, so a prediction that is confidently WRONG (it puts
          probability 0 on the true class) would send the loss to +inf and turn
          every gradient computed from it into NaN -- the network would stop
          learning with no obvious cause. Clipping the probabilities into
          [eps, 1-eps] caps the per-sample penalty at -log(eps) ~= 34.5
          instead, which is large enough to punish the mistake and still
          finite. This is why frameworks clip here rather than trusting the
          model to never output an exact 0 or 1.
    """
    m = y_pred.shape[0]
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -(y_true * np.log(y_pred)).sum() / m


def cross_entropy_prime(y_true, y_pred):
    """Cross entropy derivative

    :param numpy.array y_true: array-like of shape (n_samples,)
        Ground truth (correct) target values.
    :param numpy.array y_pred: array-like of shape (n_samples,)
        Estimated target values.
    :returns: cross entropy derivative
    """
    m = y_pred.shape[0]
    return (y_pred - y_true) / m


def softmax_cross_entropy(logits, y_true):
    """Given model outputs (logits) and the indexes
       of the true class label, computes the softmax cross entropy.

    Note: the naive form, log(sum(exp(z))), overflows. exp(1000) is inf in
          float64, so a single large logit makes the whole loss inf. The fix is
          the "log-sum-exp trick": both softmax and log-sum-exp are unchanged
          if the same constant c is subtracted from every logit in a row,

              log sum_j exp(z_j) = c + log sum_j exp(z_j - c)

          so choosing c = max_j z_j makes every exponent <= 0, hence every
          exp(...) <= 1 and finite. The result is mathematically identical --
          only the intermediate values are tamed.
    """
    logits = np.asarray(logits, dtype=float)
    true_class_logits = logits[np.arange(len(logits)), y_true]

    # c = the per-row maximum, kept 2-D so it broadcasts over the class axis
    shift = logits.max(axis=-1, keepdims=True)
    log_sum_exp = shift[:, 0] + np.log(np.sum(np.exp(logits - shift), axis=-1))
    return -true_class_logits + log_sum_exp


def softmax_cross_entropy_prime(logits, y_true):
    """Derivative of the softmax cross entropy w.r.t. the logits.

    Stabilised with the same shift as `softmax_cross_entropy`: subtracting the
    row max leaves the softmax values unchanged but keeps the exponentials
    finite.
    """
    logits = np.asarray(logits, dtype=float)
    ones_true_class = np.zeros_like(logits)
    ones_true_class[np.arange(len(logits)), y_true] = 1
    exps = np.exp(logits - logits.max(axis=-1, keepdims=True))
    softmax = exps / exps.sum(axis=-1, keepdims=True)
    return (-ones_true_class + softmax) / logits.shape[0]


def r2_score(y_true, y_pred):
    """
    R^2 regression score function.
        R^2 = 1 - SS_res / SS_tot
    where SS_res is the residual sum of squares and SS_tot is the total
    sum of squares.

    :param numpy.array y_true : array-like of shape (n_samples,) Ground truth (correct) target values.
    :param numpy.array y_pred : array-like of shape (n_samples,) Estimated target values.
    :returns: score (float) R^2 score.

    Note: R^2 answers "what fraction of the variance in y did the model
          explain?". If y_true is constant there is no variance to explain,
          SS_tot is 0 and the ratio is undefined -- the unguarded division
          returned -inf. The usual convention is used instead: a constant
          predicted exactly scores 1.0, anything else scores 0.0.
    """
    # Residual sum of squares.
    numerator = np.asarray(((y_true - y_pred) ** 2).sum(axis=0), dtype=float)
    # Total sum of squares.
    denominator = np.asarray(
        ((y_true - np.average(y_true, axis=0)) ** 2).sum(axis=0), dtype=float)

    # Guard the degenerate SS_tot == 0 case before dividing. `np.where`
    # evaluates both branches, so the denominator is patched to 1 first to
    # avoid a genuine division by zero (and its warning) in the unused branch.
    degenerate = denominator == 0
    score = np.where(degenerate,
                     np.where(numerator == 0, 1.0, 0.0),
                     1 - numerator / np.where(degenerate, 1.0, denominator))
    # single-output R^2 is a scalar; multi-output keeps one score per column
    return score if score.ndim else float(score)


METRICS = {'MSE': (mse, mse_prime),
           'RMSE': (rmse, rmse_prime),
           'MAE': (mae, mae_prime),
           'cross-entropy': (cross_entropy, cross_entropy_prime),
           'softmax-cross-entropy': (softmax_cross_entropy, softmax_cross_entropy_prime)
           }
