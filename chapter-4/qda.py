#!/usr/bin/env python3

import numpy as np


def classify(X, prior_prob, class_means, class_vars):
    '''
    Classify each data point in X
    In practice last 3 parameters are estimated using est_params

    Parameters
    ----------
    X: Numpy array of shape (N, p) consisting of input data

    prior_prob: Numpy array of shape (K, )
                Prior probabilities for each class

    class_means: Numpy array of shape (K, p)co
                 kth row is population mean of class k

    class_vars: Numpy array of shape (K, p, p)
                kth entry is covariance matrix of class k

    Returns
    -------
    y_est: Numpy array of shape (N, ) with entries in range(K)
           ith entry gives estimated class for ith input
    '''
    # Apply discriminant functions
    delta = discriminant_func(X, prior_prob, class_means, class_vars)

    y_est = np.argmax(delta, axis=1)

    return y_est


def classification_error(y, y_est):
    '''
    Returns classification error in estimating y by y_est

    Parameters
    ----------
    y: Numpy array of shape (N,) with entries 0,..,K-1
       True class outputs

    y_est: Numpy array of shape (N,) with entries 0,..,K-1
           Estimated class outputs

    Returns
    -------
    err: Float between 0 and 1
         Number of misclassified data points over total number
    '''
    N = y.shape[0]
    err = np.sum((y != y_est).astype(int)) / N

    return err


def est_params(X, Y):
    '''
    Estimate prior probabilities, class means, and class variances
    for quadratic discriminant analysis.

    Parameters
    ----------
    X: Numpy array of shape (N, p) containing input data

    Y: Numpy array of shape (N, K), indicator response matrix

    Returns
    -------
    prior_prob: Numpy array of shape (K, )
                Estimates of prior probabilities of each class

    sample_means: Numpy array of shape (K, p)
                  kth row is sample mean of class k

    sample_vars: Numpy array of shape (K, p, p)
                 kth entry is sample covariance matrix of class k
    '''
    N, p = X.shape
    _, K = Y.shape

    prior_prob = np.sum(Y, axis=0) / np.sum(Y)

    sample_means = np.empty(shape=(K, p))
    sample_vars = np.empty(shape=(K, p, p))

    for k in range(K):
        class_inputs = X[Y[:, k] == 1, :]

        sample_means[k] = np.mean(class_inputs, axis=0)

        sample_vars[k] = (class_inputs - sample_means[k]).T @ (class_inputs - sample_means[k])
        sample_vars[k] /= class_inputs.shape[0] - K

    return prior_prob, sample_means, sample_vars


def discriminant_func(X, prior_prob, class_means, class_vars):
    '''
    Apply all quadratic discriminant functions to input data X.
    In practice last 3 parameters are estimated using est_params

    Parameters
    ----------
    X: Numpy array of shape (N, p) consisting of input data

    prior_prob: Numpy array of shape (K, )
                Prior probabilities for each class

    class_means: Numpy array of shape (K, p)
                 kth row is population mean of class k

    class_vars: Numpy array of shape (K, p, p)
                kth entry is covariance matrix of class k

    Returns
    -------
    delta: Numpy array of shape (N, K)
           ith row gives K discriminant funcs applied to ith data point

    '''
    K = prior_prob.shape[0]

    # Get eigendecomposition of each covariance matrix
    eval, evect = np.linalg.eigh(class_vars)

    # Calculate D^(-1/2) for each set of evalues
    diag = np.apply_along_axis(np.diag, axis=1, arr=np.sqrt(1/eval))

    # Initialise K copies of data to apply K disc func
    delta = np.array([X] * K)

    # Calculate Mahalanobis distance for each class
    delta = delta - class_means[:, np.newaxis, :]
    delta = delta @ evect @ diag
    delta = np.sum(delta**2, axis=2).T

    # Add constant terms
    delta = prior_prob - delta / 2 - np.sum(np.log(eval), axis=1) / 2

    return delta


def gen_indicator_responses(y, K):
    '''
    Turn output vector with entries 0,..,K-1 into
    indicator reponse matrix

    Parameters
    ----------
    y: Numpy array of shape (N,) with entries 0,..,K-1

    K: Int, number of classes

    Returns
    -------
    Y: Numpy array of shape (N, K)
       1s in position [i, y[i]], 0s elsewhere
    '''
    N = y.shape[0]

    Y = np.zeros(shape=(N, K))
    Y[range(N), y] = 1

    return Y


def encode_classes(y, class_to_code):
    '''
    Replace class labels with codes
    '''
    return np.array([class_to_code[i] for i in y])


def decode_classes(y, code_to_class):
    '''
    Replace codes with class labels
    '''
    return np.array([code_to_class[i] for i in y])


def gen_class_codes(y):
    '''
    Code distinct entries of y as 0,..,K-1. Ordered with floats/ints
    first followed by strings.

    Parameters
    ----------
    y: Numpy array of size (N, ) consisting of K distinct class labels
       Entries must be floats of strings

    Returns
    -------
    code_to_class: dict with keys 0,..,K-1 and values class labels

    class_to_code: dict with class labels as keys and values 0,..,K-1
                   Inverse of code_to_class
    '''
    # Get ordered class lables
    sorted_values = np.sort(np.unique(y))

    # Generate dictionaries
    code_to_class = dict(enumerate(sorted_values))
    class_to_code = {v: k for k, v in code_to_class.items()}

    return code_to_class, class_to_code


if __name__ == '__main__':
    pass
