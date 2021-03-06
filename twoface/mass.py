""" Utilities for estimating stellar masses of APOGEE primary stars """

# Standard library

# Third-party
from astropy.utils.misc import isiterable
from astropy.constants import G
import astropy.units as u
import numpy as np
from scipy.optimize import root

# Project
from .log import log as logger

__all__ = ['CM_NM_to_CNM', 'CM_NM_to_CN', 'get_martig_vec', 'mf', 'm2_func',
           'get_m2_min', 'stellar_radius', 'period_at_surface', 'asini',
           'a2sini']

# For abundance transformations:
log_eC = 8.39
log_eN = 7.78

# Upper-triangle matrix from Martig et al.
M = np.array([[  95.87,  -10.4 ,   41.36,   15.05,  -67.61, -144.18,   -9.42],
              [   0.  ,   -0.73,   -5.32,   -0.93,    7.05,    5.12,    1.52],
              [   0.  ,    0.  ,  -46.78,  -30.52,  133.58,  -73.77,   16.04],
              [   0.  ,    0.  ,    0.  ,   -1.61,   38.94,  -15.29,    1.35],
              [   0.  ,    0.  ,    0.  ,    0.  ,  -88.99,  101.75,  -18.65],
              [   0.  ,    0.  ,    0.  ,    0.  ,    0.  ,   27.77,   28.8 ],
              [   0.  ,    0.  ,    0.  ,    0.  ,    0.  ,    0.  ,   -4.1 ]])

def CM_NM_to_CNM(M_H, C_M, N_M):
    """Compute [(C+N)/M] using Keith's method.

    Parameters
    ----------
    M_H : numeric
        [M/H], metallicity.
    C_M : numeric
        [C/M], carbon abundance.
    N_M : numeric
        [N/M], nitrogen abundance.
    """

    C_H = C_M + M_H
    N_H = N_M + M_H

    N_C = 10**(C_H + log_eC)
    N_N = 10**(N_H + log_eN)
    N_CN = 10**log_eC + 10**log_eN

    CN_H = np.log10(N_C + N_N) - np.log10(N_CN)
    return CN_H - M_H


def CM_NM_to_CN(C_M, N_M):
    """Compute [C/N].

    TODO: is this right?

    Parameters
    ----------
    C_M : numeric
        [C/M], carbon abundance.
    N_M : numeric
        [N/M], nitrogen abundance.
    """
    return (C_M + log_eC) - (N_M + log_eN)


def get_martig_vec(Teff, logg, M_H, C_M, N_M):
    """Produces a 1D vector that can be inner-producted with the upper-triangle
    matrix provided by Martig et al. to estimate the stellar mass.
    """
    vec = np.ones(7)
    vec[1] = M_H
    vec[2] = C_M
    vec[3] = N_M
    vec[4] = CM_NM_to_CNM(M_H, C_M, N_M)
    vec[5] = Teff / 4000.
    vec[6] = logg

    return vec


def mf(P, K, e):
    mf_circ = P * K**3 / (2*np.pi * G)
    return (mf_circ * (1 - e**2)**1.5).to(u.Msun)


def m2_func(m2, m1, sini, mf):
    return (m2*sini)**3 / (m1 + m2)**2 - mf


@u.quantity_input(m1=u.Msun, mf=u.Msun)
def get_m2_min(m1, mf):
    mf = mf.to(m1.unit)
    if isiterable(m1) and isiterable(mf):
        m2s = []
        for x, y in zip(m1, mf):
            try:
                res = root(m2_func, x0=10., args=(x.value, 1., y.value))
                if not res.success:
                    raise RuntimeError('Unsuccessful')
                m2s.append(res.x[0])
            except Exception as e:
                logger.debug('Failed to compute m2_min for sample: {0}'
                             .format(str(e)))
                m2s.append(np.nan)
        return m2s * m1.unit

    else:
        res = root(m2_func, x0=10., args=(m1.value, 1., mf.value))

        if res.success:
            return res.x[0] * m1.unit
        else:
            return np.nan * m1.unit


def stellar_radius(logg, mass):
    return np.sqrt(G*mass / (10**logg*u.cm/u.s**2)).to(u.Rsun)


def period_at_surface(M1, logg, e, M2=0*u.Msun):
    R1 = np.sqrt(G*M1 / (10**logg * u.cm/u.s**2))
    P = 2*np.pi * (G*(M1+M2) / R1**3)**(-1/2) * (1-e)**(-3/2)
    return P.to(u.day)


def asini(P, e, K, m1, m2):
    return (P/(2*np.pi) * np.sqrt(1-e**2) * K * (m1+m2) / m2).to(u.au)


def a2sini(P, e, K, m1, m2):
    return (P/(2*np.pi) * (m1/m2) * np.sqrt(1-e**2) * K).to(u.au)
