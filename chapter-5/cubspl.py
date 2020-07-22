#!/usr/bin/env python3

import numpy as np


def gen_nat_cubic_spl_basis_fun(knots):
    '''
    Generate a basis function for natural cubic splines at given knots. We use
    the basis N_k(x) obtained from the truncated power series basis on p.145
    of ESL.

    Parameters
    ----------
    knots: ndarray of shape (K, )
        Positions of knots

    Returns
    -------
    nat_cubic_spl_basis_fun: function float -> ndarray of shape (K, )
        Evaluate each spline in basis at input
    '''
    K = len(knots)
    trunc_power_basis = [gen_trunc_power_fun(knot, order=3) for knot in knots]

    # Reduce truncated powers to get natural cubic spline basis
    def scaled_fun_diff(fun1, fun2, scale):
        return lambda x: (fun1(x) - fun2(x)) / scale

    red_trunc_power_basis = [scaled_fun_diff(trunc_power_basis[i], trunc_power_basis[-1], knots[-1] - knots[i]) for i in range(K-1)]
    nat_trunc_power_basis = [scaled_fun_diff(red_trunc_power_basis[i], red_trunc_power_basis[-1], scale=1) for i in range(K-2)]

    def nat_cubic_spl_basis_fun(x):
        powers = np.array([1, x])
        trunc_powers = np.array([spl(x) for spl in nat_trunc_power_basis])

        return np.concatenate([powers, trunc_powers])

    return nat_cubic_spl_basis_fun


def gen_cubic_spl_basis_fun(knots):
    '''
    Generate a basis function for cubic splines at given knots. We use the
    truncated power series basis.

    Parameters
    ----------
    knots: ndarray of shape (K, )
        Positions of knots

    Returns
    -------
    nat_cubic_spl_basis_fun: function float -> ndarray of shape (K+4, )
        Evaluate each spline in basis at input
    '''
    trunc_power_basis = [gen_trunc_power_fun(knot, order=3) for knot in knots]

    def cubic_spl_basis_fun(x):
        powers = np.array([1, x, x**2, x**3])
        trunc_powers = np.array([spl(x) for spl in trunc_power_basis])

        return np.concatenate([powers, trunc_powers])

    return cubic_spl_basis_fun


def gen_trunc_power_fun(knot, order=3):
    '''
    Generate truncated power function of order at knot

    Parameters
    ----------
    knot: float
        Positions of knot

    Returns
    -------
    trunc_power: function float -> float
    '''
    def trunc_power(x):
        if x > knot:
            return (x - knot) ** order
        else:
            return 0

    return trunc_power


if __name__ == '__main__':
    pass
