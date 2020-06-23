#!/usr/bin/env python3

import numpy as np


def forward_stepwise_selection(X, y):
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
                Note: numbering is offset by 1 from input X due to bias feature

    coef: Numpy array of shape (p+1, p+2)
          kth column is approximation to coefficient vector with subset size k

    RSS: Numpy array of shape (p+2, )
         kth entry is residual sum-of-squares with subset size k
    '''
    N, p = X.shape

    # Add bias column to X
    bias = np.ones(shape=(N, 1))
    X = np.concatenate((bias, X), axis=1)

    # Initialise variables
    RSS = [y.T @ y]
    active_set = []
    inactive_set = list(range(p+1))
    coef = np.zeros(shape=(p+1, 1))

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

        # Scale orthogonal variables to norm 1 and scale base_change to match
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

    return active_set, coef, np.array(RSS)


def backward_stepwise_selection(X, y):
    '''
    Performs backward-stepwise regression assuming X has full column rank.
    We assume X doesn't have bias column so number of steps is one more than
    number of features.

    Parameters:
    ----------
    X : Numpy array of shape (N, p) containing input data

    y: Numpy array of shape (N, ) containing output data

    Returns:
    -------
    inactive_set: list of indices 0,.., p in order they were removed from model
                Note: numbering is offset by 1 from input X due to bias feature

    coef: Numpy array of shape (p+1, p+2)
          kth column is approximation to coefficient vector with subset size k

    RSS: Numpy array of shape (p+2, )
         kth entry is residual sum-of-squares with subset size k
    '''
    N, p = X.shape

    # Add bias column to X
    bias = np.ones(shape=(N, 1))
    X = np.concatenate((bias, X), axis=1)

    # Store QR decomposition of X
    Q, R = np.linalg.qr(X)
    R_inv = np.linalg.inv(R)

    # Initialise variables
    active_set = list(range(p+1))
    inactive_set = []
    coef = np.zeros(shape=(p+1, p+2))
    RSS = [y.T @ np.linalg.matrix_power(np.identity(N) - Q @ Q.T, 2) @ y]

    # In each step choose which variable to remove from active set
    for k in range(p+1)[::-1]:
        # Calculate approximation to coefficient vector at this step
        coef[active_set, k+1] = R_inv @ Q.T @ y

        # Variable with lowest score will be removed
        scores = (coef[active_set, k+1] ** 2) / np.sum(R_inv ** 2, axis=1)
        removal_index = np.argmin(scores)

        # Update values
        RSS.insert(0, RSS[0] + (Q[:, removal_index].T @ y) ** 2)

        # Remove columns (Q) and rows and columns (R_inv) for removed variable
        Q = np.delete(Q, removal_index, axis=1)
        R_inv = np.delete(R_inv, removal_index, axis=0)
        R_inv = np.delete(R_inv, removal_index, axis=1)

        inactive_set += [active_set[removal_index]]
        active_set.pop(removal_index)

    return inactive_set, coef, np.array(RSS)


def test_error(coef, X, y):
    '''
    Gives mean-squared error for different coefficient vectors against test
    data, for example where coef comes from subset selection

    Parameters:
    ----------

    coef: Numpy array of shape (p+1, n_models)
          Each column of coef is an approximation to the coefficient vector
          In stepwise selection n_models = p+1

    X: Numpy array of shape (N, p) containing test input data

    y: Numpy array of shape (N, ) containing test output data

    Returns:
    -------

    MSE: Numpy array of shape (n_models, )
         Gives MSE for columns of coef applied to test data
    '''
    # Add bias column to X
    N = X.shape[0]
    bias = np.ones(shape=(N, 1))
    X = np.concatenate((bias, X), axis=1)

    # Create matrix with all columns equal to y
    n_models = coef.shape[1]
    Y = np.array([y] * n_models).T

    MSE = np.sum((Y - X @ coef) ** 2, axis=0) / N

    return MSE


def count_inversions(list1, list2):
    '''
    Counts the number of inversions between two lists with the same elements.
    Use to compare active_set and reverse of inactive_set
    '''
    n_inv = 0

    for idx, item1 in enumerate(list1[:-1]):
        for item2 in list1[idx + 1:]:
            if list2.index(item1) > list2.index(item2):
                n_inv += 1

    return n_inv


if __name__ == '__main__':
    pass
