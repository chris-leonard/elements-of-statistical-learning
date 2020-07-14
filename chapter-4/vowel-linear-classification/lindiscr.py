#!/usr/bin/env python3

import numpy as np


def optimize_intercept_seed(X, y, coef, seed_ranges, num_tests, grid_density=10, err_tol=0.01):
    '''
    Run optimize_intercept with randomized initial_values to find
    intercept that globally minimizes classification error.

    Parameters
    ----------
    X: ndarray of shape (N, p)
        Input data to optimise against

    y: ndarray of shape (p, ), entries in 0,..,K-1
        Output data to optimise against

    coef: ndarray of shape (K, p)
        Coefficients for linear discriminant functions

    seed_ranges: list of K-1 tuples of length 2
        Pairs (low, high) to seed initial values of intercepts for each feature

    num_tests: int
        Number of tests to run with different initial_values

    grid_density: float
        Density of intercept values on each test

    err_tol: float
        Each test ends once difference between successive errors falls below

    Returns
    -------
    errs: ndarray of shape (num_tests, )
        Minimum classification errors tests

    intercepts: ndarray of shape (K, num_tests)
        Best intercepts from tests

    seeds: ndarray of shape (num_test, )
        Seed values for initial_values of intercepts

    '''
    # Calculate useful constants
    N, p = X.shape
    K = coef.shape[0]

    # Calculate gradient for reuse
    gradient = X @ coef.T

    # Initialise arrays for results
    seeds = np.empty(shape=(K, num_tests))
    intercepts = np.empty(shape=(K, num_tests))
    errs = np.empty(shape=num_tests)

    for test in range(num_tests):
        # Generate random seed
        seed = np.array([(high - low) * np.random.random() + low for low, high in seed_ranges])
        seed = np.append(seed, 0)

        # Run optimization with seed
        kwargs = {
            'X': X,
            'y': y,
            'coef': coef,
            'initial_value': seed,
            'gradient': gradient,
            'grid_density': grid_density,
            'err_tol': err_tol
        }
        seed_errs, seed_intercepts = optimize_intercept(**kwargs)

        # Store
        seeds[:, test] = seed
        errs[test] = seed_errs[-1]
        intercepts[:, test] = seed_intercepts[:, -1]

    return errs, intercepts, seeds


def optimize_intercept(X, y, coef, initial_value, gradient=None, grid_density=10, err_tol=0.01):
    '''
    Find local minimum of classification error by optimizing intercept for
    linear discriminant function from inital value.

    Parameters
    ----------
    X: ndarray of shape (N, p)
        Input data to optimise against

    y: ndarray of shape (p, ), entries in 0,..,K-1
        Output data to optimise against

    coef: ndarray of shape (K, p)
        Coefficients for linear discriminant functions

    initial_value: ndarray of shape (K, )
        Initial value of intercept

    gradient: ndarray of shape (N, K)
        Optional: stored value of X @ coef.T for performance

    grid_density: float
        Density of intercept values to test

    err_tol: float
        Algorithm ends once difference between successive errors falls below

    Returns
    -------
    errs: ndarray of shape (num_loops+1, )
        Classification errors against given data for successive intercepts

    intercepts: ndarray of shape (K, num_loops+1)
        Intercepts from steps of algorithm
    '''
    # Calculate useful constants
    N, p = X.shape
    K = coef.shape[0]

    # Calculate gradient for reuse
    if gradient is None:
        gradient = X @ coef.T

    # Initialise values
    intercept = initial_value
    err = classification_error(intercept, gradient, y)
    err_change = np.inf

    # Store intercepts and errors in an array
    intercepts = initial_value[:, np.newaxis]
    errs = np.array([err])

    # For performance monitoring
    num_loops = 0

    while err_change >= err_tol:
        num_loops += 1

        # Optimise intercept for each class with other intercepts constant
        for cls in range(K-1):
            # Calculate log odds against class for each data point
            log_odds = np.delete(gradient, cls, axis=1) - gradient[:, cls][:, np.newaxis]
            log_odds += np.delete(intercept, cls)

            # We classify to the feature with greatest log odds
            log_odds_max = np.max(log_odds, axis=1)

            # Range of values to try - class intercept will vary from
            # when we never classify to feat to when we always do
            cls_range_max = np.max(log_odds_max)
            cls_range_min = np.min(log_odds_max)
            cls_range = np.arange(cls_range_min, cls_range_max, 1/grid_density)
            M = len(cls_range)

            # Create grid of intercepts to try
            intercept_grid = np.array([intercept] * M)
            intercept_grid[:, cls] = cls_range

            # Calculate error for each set of intercepts
            err = classification_error(intercept_grid, gradient, y)

            # Best values
            min_err = np.min(err)
            intercept = intercept_grid[np.argmin(err), :]

        # Record error and intercept
        err_change = np.abs(errs[-1] - min_err)
        errs = np.append(errs, min_err)
        intercepts = np.concatenate([intercepts, intercept[:, np.newaxis]], axis=1)

    return errs, intercepts


def classification_error(intercept, gradient, y):
    '''
    Calculate classification error for linear discriminant functions against
    output data y.

    Parameters
    ----------
    intercept: ndarray of shape (K, ) or (M, K)
        Either a single intercept or M different intercepts

    gradient: ndarray of shape (N, K)
        X @ coef.T for input data X

    y: ndarray of shape (N, ), values in 0,..,K-1
        True outputs

    Returns
    -------
    err: float or ndarray of shape (M, )
        Classification errors against y
    '''
    N = len(y)

    if intercept.ndim == 1:
        y_pred = np.argmax(intercept + gradient, axis=1)
        err = np.sum((y_pred != y).astype(int)) / N

    elif intercept.ndim == 2:
        y_pred = intercept[:, np.newaxis, :] + gradient
        y_pred = np.argmax(y_pred, axis=2)
        err = np.sum((y_pred != y).astype(int), axis=1) / N

    return err


if __name__ == '__main__':
    pass
