#!/usr/bin/env python3

import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize
from scipy.stats import norm
from cubspl import gen_nat_cubic_spl_basis_fun


def run_sim(reg_fun, eff_df, alpha, sample_range, N=100, sigma=1):
    '''
    Generates data sample, fits smoothing splines, and plots regression
    function f(x) against pointwise mean of fhat(x) and quantiles with given
    alpha value.

    Parameters
    ----------
    reg_fun: univariate function that can operate on arrays
        Regression function f(x)

    eff_df: int between 1 and N
        Effective degrees of freedom for smoothing splines

    alpha: float between 0 and 1
        Probability an observation lies outside quantiles

    sample_range: tuple of length 2
        Range from which to sample x-values

    N: int
        Number of samples

    sigma: float
        Standard deviation of Y
    '''
    # Generate sample data for simulation
    X, f, y = gen_data(reg_fun, sample_range, N, sigma)
    print('Sampling {} points uniformly on [{}, {}]'.format(N, sample_range[0], sample_range[1]))
    print('')

    # Generate matrix of natural splines at knots X, evaluated at X
    basis_fun = gen_nat_cubic_spl_basis_fun(knots=X)
    basis_matrix = np.array([basis_fun(x) for x in X])
    basis_matrix_inv = np.linalg.inv(basis_matrix)

    # Generate matrix Omega
    Omega = gen_Omega(knots=X)

    # Penalty matrix K
    penalty_matrix = basis_matrix_inv.T @ Omega @ basis_matrix_inv

    # Find lambda value corresponding to degrees of freedom
    lam = df_to_lambda(eff_df, penalty_matrix=penalty_matrix)
    print('Effective degrees of freedom: {}'.format(eff_df))
    print('Smoothing parameter lambda: {:.5f}'.format(lam))
    print('')

    # Calculate smoother matrix
    smoother_matrix = gen_smoother_matrix(lam, basis_matrix, Omega)

    # x-values for plot
    x_val = np.linspace(sample_range[0], sample_range[1], num=1000)

    # Mean and quantiles for plotting
    mean_curve = [fhat_mean(x, f, basis_fun, smoother_matrix, basis_matrix_inv) for x in x_val]
    quant_curves = [fhat_quantiles(x, alpha, f, sigma, basis_fun, smoother_matrix, basis_matrix_inv) for x in x_val]
    lower_quant_curve = [pair[0] for pair in quant_curves]
    upper_quant_curve = [pair[1] for pair in quant_curves]
    print('Plotting {}% quantiles:'.format(round(100*(1-alpha), 1)))

    # Set plot parameters
    fig, ax = plt.subplots(figsize=(15, 10))

    ax.set_title('Mean of Approximation to Regression Function with Quantiles', fontsize=20)

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # Plot regression function and sample values
    ax.plot(x_val, reg_fun(x_val), color='k', label='regression_function')
    ax.scatter(X, y, marker='o', facecolors='none', color='k')

    # Plot mean and quantiles of fhat
    ax.plot(x_val, mean_curve, color='red', label='fhat_mean')
    ax.plot(x_val, lower_quant_curve, color='blue', ls='--', label='fhat_quantiles')
    ax.plot(x_val, upper_quant_curve, color='blue', ls='--')

    ax.legend(fontsize='x-large')
    plt.show()


def fhat_mean(x, f, basis_fun, smoother_matrix, basis_matrix_inv):
    '''
    Pointwise mean of approximation to f at input x.

    Parameters
    ----------
    x: float
        Point at which mean is calculated

    f: ndarray of shape (N,)
        Regression function applied to sample inputs X

    basis_fun: function x -> ndarray of shape (N,)
        Evaluate natural cubic spline basis at x

    smoother_matrix: ndarray of shape (N, N)
        Smoother matrix S_lambda

    basis_matrix_inv: ndarray of shape (N, N)
        Inverse of the basis matrix N

    Returns
    -------
    mean: float
        Pointwise mean of fhat at x
    '''
    A = basis_fun(x) @ basis_matrix_inv @ smoother_matrix
    mean = A @ f

    return mean


def fhat_quantiles(x, alpha, f, sigma, basis_fun, smoother_matrix, basis_matrix_inv):
    '''
    Upper and lower alpha quantiles of approximation to f at input x.

    Parameters
    ----------
    x: float
        Point at which quantiles are calculated

    alpha: float between 0 and 1
        Probability an observation lies outside quantiles

    f: ndarray of shape (N,)
        Regression function applied to sample inputs X

    sigma: positive float
        Standard deviation of Y

    basis_fun: function x -> ndarray of shape (N,)
        Evaluate natural cubic spline basis at x

    smoother_matrix: ndarray of shape (N, N)
        Smoother matrix S_lambda

    basis_matrix_inv: ndarray of shape (N, N)
        Inverse of the basis matrix N

    Returns
    -------
    quantiles: tuple of two floats
        Pointwise upper and lower alpha quantiles at x
    '''
    # Calculate mean and standard deviation of fhat
    A = basis_fun(x) @ basis_matrix_inv @ smoother_matrix
    fhat_mean = A @ f
    fhat_stdev = sigma * np.linalg.norm(A)

    # Calculate alpha quantiles using normal distribution
    z = norm.ppf(1 - alpha / 2)

    lower_quant = fhat_mean - fhat_stdev * z
    upper_quant = fhat_mean + fhat_stdev * z

    return [lower_quant, upper_quant]


def df_to_lambda(n, penalty_matrix=None, eigval=None, upper_limit=1):
    '''
    Calculate lambda value corresponding to effective degrees of freedom.
    Requires either penalty_matrix or eigenvalues.

    Parameters
    ----------
    n: int between 1 and N
        Lambda value

    penalty_matrix: ndarray of shape (N, N)
        Penalty matrix K

    eigval: ndarray of shape (N,)
        Eigenvalues of penalty matrix

    upper_limit: positive float
        We solve for lambda values in [0, upper_limit]

    Returns
    -------
    lam: positive float
        Lambda value corresponding to effective degrees of freedom n
    '''
    # Get eigenvalues of penalty matrix
    if penalty_matrix is None and eigval is None:
        raise TypeError('Need penalty_matrix or eigval argument')
    elif eigval is None:
        eigval, _ = np.linalg.eig(penalty_matrix)

    # Function to find roots of
    def F(x):
        return lambda_to_df(x, eigval=eigval) - n

    # Find roots of F
    try:
        lam = optimize.brentq(F, 0, upper_limit)
    except ValueError:
        print('Could not find lambda value. Increase upper_limit or change n')

    return lam


def lambda_to_df(lam, penalty_matrix=None, eigval=None):
    '''
    Calculate effective degrees of freedom for lambda value. Requires either
    penalty_matrix or eigenvalues

    Parameters
    ----------
    lam: positive float
        Lambda value

    penalty_matrix: ndarray of shape (N, N)
        Penalty matrix K

    eigval: ndarray of shape (N,)
        Eigenvalues of penalty matrix

    Returns
    -------
    eff_df: int between 1 and N
        Effective degrees of freedom corresponding to lambda
    '''
    # Get eigenvalues of penalty matrix
    if penalty_matrix is None and eigval is None:
        raise TypeError('Need penalty_matrix or eigval argument')
    elif eigval is None:
        eigval, _ = np.linalg.eig(penalty_matrix)

    return np.sum(1 / (1 + lam * eigval))


def gen_Omega(knots):
    '''
    Calculate the matrix Omega_N of integrals of products of second
    derivatives of elements of the natural cubic spline basis at given knots.

    Parameters
    ----------
    knots: ndarray of shape (N,)
        Locations of knots for natural cubic splines

    Returns
    -------
    Omega: ndarray of shape (N, N)
        (j,k) entry is integral of Nj''(t)* Nk''(t)
    '''
    N = len(knots)

    def Omega_jk(j, k, knots):
        'Entries of Omega coming from truncated power basis elements'
        upper_idx = max(j, k)
        lower_idx = min(j, k)

        # This formula comes from explicit computation
        omega = 12*(knots[-2] - knots[lower_idx])*(knots[-2] - knots[upper_idx]) / (knots[-1] - knots[lower_idx])

        if upper_idx != lower_idx:
            omega += 6*(knots[-2] - knots[upper_idx])**2 / (knots[-1] - knots[upper_idx])
            omega -= 6*(knots[-2] - knots[upper_idx])**2 / (knots[-1] - knots[lower_idx])

        return omega

    # First 2 rows/columns are zero due to linear basis terms
    Omega = np.zeros(shape=(N, N))
    Omega[2:, 2:] = np.array([[Omega_jk(j, k, knots=knots) for j in range(N-2)] for k in range(N-2)])

    return Omega


def gen_smoother_matrix(lam, basis_matrix, Omega):
    '''
    Generate smoother matrix S_lambda.

    Parameters
    ----------
    lam: positive float
        Smoothing parameter lambda

    basis_matrix: ndarray of shape (N, N)
        Basis matrix N

    Omega: ndarray of shape (N, N)
        Matrix Omega_N

    Returns
    -------
    smoother_matrix: ndarray of shape (N, N)
    '''
    return basis_matrix @ np.linalg.inv(basis_matrix.T @ basis_matrix + lam * Omega) @ basis_matrix.T


def gen_data(reg_fun, sample_range, N=100, sigma=1):
    '''
    Generate sample data for simulation. We sample N x-values uniformly from
    sample_range, apply regression function, and add normally distributed
    errors with standard deviation sigma to get y.

    Parameters
    ----------
    reg_fun: univariate function that can operate on arrays
        Regression function f(X)

    sample_range: tuple of length 2
        Range from which to sample X-values

    N: int
        Number of samples

    sigma: float
        Standard deviation of Y

    Returns
    -------
    X: ndarray of shape (N,)
        Input data, in increasing order

    f: ndarray of shape (N,)
        Regression function applied to X

    y: ndarray of shape (N,)
        Outputs - f + random error
    '''
    # Sample input values from uniform distribution
    X = np.random.uniform(low=sample_range[0], high=sample_range[1], size=(N,))
    X = np.sort(X)

    # Apply regression function and add random error
    f = reg_fun(X)
    y = f + np.random.normal(loc=0, scale=sigma, size=(N,))

    return X, f, y


if __name__ == '__main__':
    eff_df = 8
    alpha = 0.05
    sample_range = (0, 1)
    N = 100
    sigma = 1

    def reg_fun(x):
        return np.sin(12 * (x + 0.2)) / (x + 0.2)

    run_sim(reg_fun, eff_df, alpha, sample_range, N, sigma)
