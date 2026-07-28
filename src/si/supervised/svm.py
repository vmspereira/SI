# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Created By  : Vítor Pereira
# Created Date: 01-09-2021
# version ='0.0.1'
# ---------------------------------------------------------------------------
"""Support Vector Machine module"""
# ---------------------------------------------------------------------------
# A Support Vector Machine (SVM) is a binary classifier (labels +1 / -1) that
# looks for the decision boundary with the *largest margin* — the widest gap
# between the two classes. Maximising the margin tends to generalise well.
#
# The neat part is the *kernel trick*. Linearly inseparable data can often be
# separated by a hyperplane after mapping it into a higher-dimensional space.
# A kernel K(x1, x2) computes the inner product in that space *without* ever
# building the mapping explicitly. Because the SVM's training problem and its
# decision function depend on the data only through such inner products, we can
# swap the plain dot product for any valid kernel and get a non-linear boundary
# for free.
#
# This implementation solves the *dual* form of the soft-margin SVM as a
# Quadratic Program (QP) with cvxopt. The dual variables are the Lagrange
# multipliers (alpha), one per training sample. The handful of samples with
# alpha > 0 are the *support vectors* — the only points that define the
# boundary. C is the soft-margin penalty: it bounds the alphas (0 <= alpha <= C)
# and trades off a wider margin against tolerating misclassified points.
#
# Use an SVM for medium-sized datasets where a clear margin exists; with an RBF
# kernel it is a strong non-linear classifier with few hyper-parameters.
# ---------------------------------------------------------------------------

import numpy as np
from .model import Model
from si.util import accuracy_score
from cvxopt import matrix, solvers


# Each kernel below is a closure: given its parameters it returns a function
# f(x1, x2) that computes the kernel value (an inner product in the implicit
# feature space) for a pair of samples.

def linear_kernel(**kwargs):
    # No feature-space mapping: K(x1, x2) = x1 . x2. Gives a linear boundary.
    def f(x1, x2):
        return np.inner(x1, x2)
    return f


def polynomial_kernel(power, coef, **kwargs):
    # K(x1, x2) = (x1 . x2 + coef)^power. Implicitly maps to all feature
    # products up to degree `power`, yielding a polynomial decision boundary.
    def f(x1, x2):
        return (np.inner(x1, x2) + coef)**power
    return f


def rbf_kernel(gamma, **kwargs):
    # Gaussian / RBF kernel: K(x1, x2) = exp(-gamma * ||x1 - x2||^2). It is a
    # similarity that decays with distance (1 when identical, -> 0 when far
    # apart); gamma sets how quickly. Corresponds to an infinite-dimensional
    # feature space, so it can fit very flexible, non-linear boundaries.
    def f(x1, x2):
        distance = np.linalg.norm(x1 - x2) ** 2
        return np.exp(-gamma * distance)
    return f


class SVM(Model):
    """The Support Vector Machine classifier.
    Uses cvxopt to solve the quadratic optimization problem.
    Adapted from https://github.com/eriklindernoren/ML-From-Scratch.

    :param float C: Penalty term.
    :param callable kernel: Kernel function. Can be either polynomial, rbf or linear.
    :param int power: The degree of the polynomial kernel. Will be ignored by the other
        kernel functions.
    :param float gamma: Used in the rbf kernel function.
    :param float coef: Bias term used in the polynomial kernel function.
    """
    # `predict` takes a 2-D X (n_samples, n_features) and returns one
    # prediction per row -- see the note on Model.predicts_batch.
    predicts_batch = True

    def __init__(self, C=1, kernel=rbf_kernel, power=4, gamma=None, coef=4):
        self.C = C
        self.kernel = kernel
        self.power = power
        self.gamma = gamma
        self.coef = coef
        # Learned during fit():
        # lagr_multipliers      = the non-zero dual variables alpha (one per SV).
        # support_vectors       = the training samples with alpha > 0.
        # support_vector_labels = their +1/-1 labels.
        # intercept             = the bias term b of the decision function.
        self.lagr_multipliers = None
        self.support_vectors = None
        self.support_vector_labels = None
        self.intercept = None

    def fit(self, dataset):
        self.dataset = dataset
        X, y = dataset.getXy()
        n_samples, n_features = np.shape(X)

        # Set gamma to 1/n_features by default
        if not self.gamma:
            self.gamma = 1 / n_features

        # Initialize kernel method with parameters
        self.kernel = self.kernel(
            power=self.power,
            gamma=self.gamma,
            coef=self.coef)

        # Calculate kernel matrix
        # kernel_matrix[i, j] = K(x_i, x_j): all pairwise inner products in the
        # (implicit) feature space. This is the only way the data enters the QP.
        kernel_matrix = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                kernel_matrix[i, j] = self.kernel(X[i], X[j])

        # Define the quadratic optimization problem
        # The dual SVM is: maximise sum_i alpha_i - 1/2 sum_ij a_i a_j y_i y_j K(x_i,x_j)
        # subject to  sum_i a_i y_i = 0  and  0 <= a_i <= C.
        # cvxopt minimises 1/2 a^T P a + q^T a  s.t.  G a <= h, A a = b, so we
        # map the dual onto those matrices below.
        #
        # P[i,j] = y_i y_j K(x_i,x_j) is the quadratic term (outer(y,y) elementwise
        # times the kernel matrix). q = -1 turns the "+ sum a_i" into a minimisation.
        P = matrix(np.outer(y, y) * kernel_matrix, tc='d')
        q = matrix(np.ones(n_samples) * -1)
        # Equality constraint sum_i a_i y_i = 0  ->  A a = b with A = y, b = 0.
        A = matrix(y, (1, n_samples), tc='d')
        b = matrix(0, tc='d')

        if not self.C:
            # Hard margin (no penalty): only the lower bound a_i >= 0, encoded as
            # -a_i <= 0.
            G = matrix(np.identity(n_samples) * -1)
            h = matrix(np.zeros(n_samples))
        else:
            # Soft margin: enforce both 0 <= a_i (G_max: -a_i <= 0) and
            # a_i <= C (G_min: a_i <= C), stacked into one inequality system.
            G_max = np.identity(n_samples) * -1
            G_min = np.identity(n_samples)
            G = matrix(np.vstack((G_max, G_min)))
            h_max = matrix(np.zeros(n_samples))
            h_min = matrix(np.ones(n_samples) * self.C)
            h = matrix(np.vstack((h_max, h_min)))

        # Solve the quadratic optimization problem using cvxopt
        minimization = solvers.qp(P, q, G, h, A, b)

        # Lagrange multipliers
        # the optimal alpha vector (one multiplier per training sample).
        lagr_mult = np.ravel(minimization['x'])

        # Extract support vectors
        # Get indexes of non-zero lagr. multipiers
        # alpha_i is (numerically) zero for non-support points; the rest, with
        # alpha_i > 0, are the support vectors that define the boundary.
        idx = lagr_mult > 1e-7
        # Get the corresponding lagr. multipliers
        self.lagr_multipliers = lagr_mult[idx]
        # Get the samples that will act as support vectors
        self.support_vectors = X[idx]
        # Get the corresponding labels
        self.support_vector_labels = y[idx]

        # Calculate intercept with first support vector
        # On a support vector, y = sum_i a_i y_i K(x_i, x_sv) + b. Solving for b
        # using the first SV: b = y_sv - sum_i a_i y_i K(x_i, x_sv).
        self.intercept = self.support_vector_labels[0]
        for i in range(len(self.lagr_multipliers)):
            self.intercept -= self.lagr_multipliers[i] * self.support_vector_labels[
                i] * self.kernel(self.support_vectors[i], self.support_vectors[0])

    def predict(self, X):
        y_pred = []
        # Iterate through list of samples and make predictions
        for sample in X:
            prediction = 0
            # Determine the label of the sample by the support vectors
            # Decision function f(x) = sum_i a_i y_i K(x_i, x) + b, summed only
            # over the support vectors (all other alphas are 0).
            for i in range(len(self.lagr_multipliers)):
                prediction += self.lagr_multipliers[i] * self.support_vector_labels[
                    i] * self.kernel(self.support_vectors[i], sample)
            prediction += self.intercept
            # The predicted class is the sign of f(x): +1 on one side of the
            # boundary, -1 on the other.
            y_pred.append(np.sign(prediction))
        return np.array(y_pred)

    def cost(self, X=None, y=None):
        X = X if X is not None else self.dataset.X
        y = y if y is not None else self.dataset.y
        return accuracy_score(y, self.predict(X))