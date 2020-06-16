#!/usr/bin/env python3

import numpy as np


def forward_stepwise_regression(X, y):
    '''
    Performs forward-stepwise regression assuming X has full column rank.
    We assume X doesn't have bias column so number of steps is one more than
    number of features.

    Parameters:
    ----------
    X : Numpy array of shape (N, p) containing input data

    y: Numpy array of shape (N, ) containing output data

    Returns:
    -------
    active_set: list of indices 0,.., p in the order they were added to model

    coef: Numpy array of shape (p+1, p+1)
          kth column is the approximation to the coefficient vector after the kth step

    RSS: list of p+2 integers
         Residual sum-of-squares after successive steps in algorithm
    '''
    N, p = X.shape

    # Add bias column to X
    bias = np.ones(shape=(N, 1))
    X = np.concatenate((bias, X), axis=1)

    # Initialise variables
    RSS = [y.T @ y]
    active_set = []
    inactive_set = list(range(p+1))
    coef = np.empty(shape=(p+1, 0))

    # Part of QR-decomposition of X
    Q = np.empty(shape=(N, 0))
    R_inv = np.empty(shape=(p+1, 0))

    # In each step we choose which variable to add to active set
    for k in range(p+1):
        # Base change orthogonalises new variables against those in active set
        if k == 0:
            base_change = np.identity(p+1)
        else:
            base_change = np.identity(p+1) - R_inv @ Q.T @ X

        candidates = X @ base_change

        # Active variables should go to zero, this resolves precision errors
        base_change[:, active_set] = 0
        candidates[:, active_set] = 0

        # Scale orthogonalised variables to norm 1 and scale base_change to match
        candidates_norms = np.linalg.norm(candidates[:, inactive_set], axis=0)
        base_change[:, inactive_set] /= candidates_norms
        candidates[:, inactive_set] /= candidates_norms

        # Choose candidate with greatest correlation with y
        candidates_scores = (candidates.T @ y) ** 2
        best_candidate_index = np.argmax(candidates_scores)

        inactive_set.remove(best_candidate_index)
        active_set += [best_candidate_index]

        # Update values
        RSS += [RSS[-1] - candidates_scores[best_candidate_index]]

        Q = np.concatenate((Q, candidates[:, best_candidate_index][:,  None]), axis=1)
        R_inv = np.concatenate((R_inv, base_change[:, best_candidate_index][:, None]), axis=1)
        coef = np.concatenate((coef, (R_inv @ Q.T @ y)[:, None]), axis=1)

    return active_set, coef, RSS


if __name__ == '__main__':
    pass
