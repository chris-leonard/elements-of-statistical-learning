#!/usr/bin/env python3

import numpy as np
from scipy import stats
from math import sqrt
from matplotlib import pyplot as plt


def run_sim(beta, alpha, N, sigma, xmean, xstdev, plot_range=None):
    '''
    Calculates and plots confidence bounds about a regression function of the
    form f(X) = beta_0 + beta_1 X + beta_2 X^2 + beta_3 X^3. The data is
    simulated with X ~ N(xmean, xstdev^2) and Y ~ N(f(X), sigma^2). We take a
    sample of size N and calculate 100(1-alpha)% confidence bands for f using
    two methods-one based on a t statistic and one on a chi-squared statistic.
    We plot the regression function against its confidence bands for x in the
    plot_range and print a few relevant statistics.
    '''
    if plot_range is None:
        plot_range = plot_range = [xmean - 2 * xstdev, xmean + 2 * xstdev]

    # Generate sample data
    X, y = gen_sample(beta, N, sigma, xmean, xstdev)

    # Calculate least squares fit
    XTXinv = gen_XTXinv(X)
    betahat, yhat = fit_ls(X, y)
    print('Beta:', beta)
    print('Betahat:', betahat.round(3))
    print('\n')

    # Calculate RSS and variance estimate
    rss = calc_rss(y, yhat)
    sigmahat = gen_sigmahat(y, yhat)
    print('RSS: ', round(rss, 1))
    print('Standard error: ', round(sigmahat, 3))
    print('\n')

    # Generate regression functions
    regfun = gen_regfun(beta)
    regfunhat = gen_regfun(betahat)

    # Generate functions to calculate confidence bounds
    tconf = gen_tconf(regfunhat, sigmahat, X, alpha, XTXinv)
    chi2conf = gen_chi2conf(regfunhat, sigmahat, X, alpha, XTXinv)
    print('Endpoints of {}% confidence interval at xmean:'.format(round(100*(1-alpha), 1)))
    print('T method: ', [round(endpoint, 3) for endpoint in tconf(xmean)])
    print('Chi-square method: ', [round(endpoint, 3) for endpoint in chi2conf(xmean)])
    print('\n')

    # Plot regression function against confidence bounds
    plot_conf(regfun, tconf, chi2conf, plot_range)


def plot_conf(regfun, tconf, chi2conf, plot_range):
    '''
    Plots regression function against confidence bounds within plot_range
    using tconf and chi2conf functions
    '''
    # Data for plotting
    x0_range = np.linspace(plot_range[0], plot_range[1], 100)

    tconf_array = np.array([tconf(x0) for x0 in x0_range])
    tconf_lower = tconf_array[:, 0]
    tconf_upper = tconf_array[:, 1]

    chi2conf_array = np.array([chi2conf(x0) for x0 in x0_range])
    chi2conf_lower = chi2conf_array[:, 0]
    chi2conf_upper = chi2conf_array[:, 1]

    # Set plot parameters
    fig, ax = plt.subplots(figsize=(15, 10))

    ax.set_title('Regression Function with Confidence Bounds', fontsize=20)

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # Plot data
    ax.plot(x0_range, regfun(x0_range), color='black', label='regression function')

    ax.plot(x0_range, tconf_lower, color='red', ls='--', label='t conf bounds')
    ax.plot(x0_range, tconf_upper, color='red', ls='--')

    ax.plot(x0_range, chi2conf_lower, color='blue', ls='--', label='chi2 conf bounds')
    ax.plot(x0_range, chi2conf_upper, color='blue', ls='--')

    ax.legend(fontsize='x-large')

    plt.show()


def gen_tconf(regfunhat, sigmahat, X, alpha, XTXinv=None):
    if XTXinv is None:
        XTXinv = gen_XTXinv(X)

    def tconf(x0):
        N = X.shape[0]
        x0vect = np.array([1, x0, x0**2, x0**3])
        t = stats.t(df=N-4).ppf((1-alpha/2))

        int_halflength = sigmahat * sqrt(x0vect.T @ XTXinv @ x0vect) * t
        int_midpoint = regfunhat(x0)

        conf_int = [int_midpoint - int_halflength, int_midpoint + int_halflength]

        return conf_int

    return tconf


def gen_chi2conf(regfunhat, sigmahat, X, alpha, XTXinv=None):
    if XTXinv is None:
        XTXinv = gen_XTXinv(X)

    def chi2conf(x0):
        x0vect = np.array([1, x0, x0**2, x0**3])
        chi2 = stats.chi2(df=4).ppf(1-alpha)
        chi = sqrt(chi2)

        int_halflength = sigmahat * chi * (x0vect.T @ XTXinv @ x0vect) / sqrt((X @ XTXinv @ x0vect).T @ (X @ XTXinv @ x0vect))
        int_midpoint = regfunhat(x0)

        conf_int = [int_midpoint - int_halflength, int_midpoint + int_halflength]

        return conf_int

    return chi2conf


def gen_sigmahat(y, yhat):
    '''
    Calculate unbiased estimates for stdev and variance
    '''
    N = y.size
    sigma2hat = (y-yhat).T@(y-yhat)/(N-4)
    sigmahat = sqrt(sigma2hat)

    return sigmahat


def fit_ls(X, y):
    '''
    Calculate least squares fit of data.
    Returns:
    betahat: np.array of shape (4), least squares estimate of beta
    yhat: np.array of shape (N), least squares estimate of y
    '''
    betahat = np.linalg.inv(X.T@X)@X.T@y
    yhat = X@betahat

    return betahat, yhat


def gen_sample(beta, N, sigma, xmean, xstdev):
    '''
    Generates a sample of size N.
    Returns:
    X: np.array of shape (N, 4). jth column is x^j with x sampled from N(xmean, xstdev^2)
    y: np.array of shape (N) sampled from N(Xbeta, sigma^2)
    '''
    x = np.random.normal(loc=xmean, scale=xstdev, size=(N))
    X = np.expand_dims(x, axis=1)

    bias = np.ones(shape=(N, 1))
    X = np.concatenate((bias, X, X**2, X**3), axis=1)

    y = np.random.normal(loc=X@beta, scale=sigma, size=N)

    return X, y


def gen_regfun(beta):
    '''
    Generate regression function based on parameter vector beta.
    Returns:
    regfun: x -> (1, x, x^2, x^3)^T beta
    '''
    def regfun(x0):
        x0vect = np.array([1, x0, x0**2, x0**3])
        return x0vect.T@beta

    return regfun


def gen_XTXinv(X):
    'Calculate (X^T X)^(-1) for future calculations'
    XTXinv = np.linalg.inv(X.T@X)

    return XTXinv


def calc_rss(y, yhat):
    'Calculate the residual sum of squares'
    rss = (y - yhat).T @ (y - yhat)

    return rss


if __name__ == '__main__':
    beta = [0, 0, 0, 1]
    N = 10
    sigma = 1
    xmean = 0
    xstdev = 2
    alpha = 0.05

    run_sim(beta, alpha, N, sigma, xmean, xstdev, plot_range=[-1, 1])
