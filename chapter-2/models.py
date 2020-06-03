#!/usr/bin/env python3

import numpy as np
from numpy import linalg


def classification_error(X, y, model, *model_args):
    '''
    Calculates binary classification error for a model applied to X, with actual output y.

    Inputs
    ------
    X: 2-dimensional numpy array of inputs. Dimension (p, N)
    y: Numpy array of actual outputs, values 0 or 1. Dimension (p, )
    model: Function taking inputs X and *model_args

    Output
    ------
    error: Integer. Proportion of samples that were misclassified
    '''
    # Apply model to classification problem
    yhat = model(X, *model_args)
    yhat = np.around(yhat, decimals=0)

    # Calculate classification error
    p = y.size
    error = np.sum(yhat != y) / p

    return error


def apply_k_nearest_neighbour(X, Xtrain, ytrain, k):
    '''
    Approximate output from X using k-nearest neighbours with training set (Xtrain, ytrain).

    Inputs
    ------
    Xtrain: Numpy array of training inputs (with bias column). Dimension (p, N+1)
    ytrain : Numpy array of training outputs. Dimension (p, )
    '''
    # Put in 2-dimensional array
    if X.ndim == 1:
        X = np.expand_dims(X, axis=0)

    # Array of with i,j entry distance from Xtrain[i] to Xinput[j]
    dist = np.array([[train_row - input_row for input_row in X] for train_row in Xtrain])
    dist = np.sum(dist*dist, axis=2)

    # Find k closest points for each Xinput and take average of y's
    min_indices = np.argpartition(dist, k-1, axis=0)[:k, :]
    yhat = np.mean(ytrain[min_indices], axis=0)

    return yhat


def apply_linear_model(X, beta):
    '''
    Apply linear model to input array X using parameters beta

    Inputs
    ------
    X: 1 or 2-dimensional Numpy array inputs (without bias column). Dimension (p, N) or (N, )
    beta: Numpy array of model coefficients. Dimension (N+1, )

    Output
    ------
    yhat: Numpy array of predicted outputs. Dimension (p,)
    '''
    # Put in 2-dimensional array
    if X.ndim == 1:
        X = np.expand_dims(X, axis=0)

    # Add bias
    X = add_bias(X)

    # Apply
    yhat = X @ beta

    return yhat


def train_linear_model(Xtrain, ytrain):
    '''
    Trains a linear regression model using least squares.

    Inputs
    ------
    Xtrain: Numpy array inputs (without bias column). Dimension (p, N)
    ytrain: Numpy array outputs. Dimension (p, )

    Outputs
    -------
    beta: Numpy array of trained coefficients. Dimension (N+1, )
    '''
    # Add bias column
    Xtrain = add_bias(Xtrain)

    beta = linalg.inv(Xtrain.T @ Xtrain) @ Xtrain.T @ ytrain

    return beta


def add_bias(X):
    '''Adds bias column to an array of input data'''
    p, N = X.shape

    bias = np.ones(shape=(p, 1))
    X = np.concatenate((bias, X), axis=1)

    return X


if __name__ == '__main__':
    pass
