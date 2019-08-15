# Third-party
import astropy.units as u
import numpy as np
from thejoker.sampler.mcmc import TheJokerMCMCModel

from twoface.log import log as logger

__all__ = ['unimodal_P', 'max_likelihood_sample', 'MAP_sample']


def unimodal_P(samples, data):
    """Check whether the samples returned are within one period mode.

    Parameters
    ----------
    samples : `~thejoker.JokerSamples`

    Returns
    -------
    is_unimodal : bool
    """

    P_samples = samples['P'].to(u.day).value
    P_min = np.min(P_samples)
    T = np.ptp(data.t.mjd)
    delta = 4*P_min**2 / (2*np.pi*T)

    return np.ptp(P_samples) < delta


def max_likelihood_sample(data, samples):
    """Return the maximum-likelihood sample.

    Parameters
    ----------
    data : `~thejoker.RVData`
    samples : `~thejoker.JokerSamples`

    """
    chisqs = np.zeros(len(samples))

    for i in range(len(samples)):
        orbit = samples.get_orbit(i)
        residual = data.rv - orbit.radial_velocity(data.t)
        err = np.sqrt(data.stddev**2 + samples['jitter'][i]**2)
        chisqs[i] = np.sum((residual**2 / err**2).decompose())

    return samples[np.argmin(chisqs)]


def MAP_sample(data, samples, joker_params, return_index=False):
    """Return the maximum a posteriori sample.

    Parameters
    ----------
    data : `~thejoker.RVData`
    samples : `~thejoker.JokerSamples`
    joker_params : `~thejoker.JokerParams`

    """
    model = TheJokerMCMCModel(joker_params, data)

    logger.debug(np.shape(samples))

    mcmc_p = model.pack_samples(samples)
    ln_ps = np.zeros(len(mcmc_p))
    logger.debug("in MAP {0} {1}".format(np.shape(mcmc_p),len(mcmc_p)))
    for i in range(len(ln_ps)):
        logger.debug("in loop {0} {1} {2}".format(i,len(mcmc_p[i]),len(mcmc_p.T[i])))
        ln_ps[i] = model.ln_posterior(mcmc_p.T[i])

    logger.debug("done loop")

    if return_index:
        idx = np.argmax(ln_ps)
        logger.debug("returning idx")
        return samples[idx], idx
    else:
        logger.debug("returning")
        return samples[np.argmax(ln_ps)]
