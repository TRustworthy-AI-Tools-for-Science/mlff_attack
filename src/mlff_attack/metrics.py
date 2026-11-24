"""Metrics for evaluating MLFF attack outputs."""

import logging

import numpy as np
import scipy.stats

logger = logging.getLogger(__name__)


def calculate_spectral_flatness(signal):
    """Calculate the spectral flatness of a signal.

    Parameters
    ----------
    signal : array-like
        Input 1D signal array.

    Returns
    -------
    float or None
        Spectral flatness value.
    """
    try:
        signal = np.asarray(signal)
        if signal.ndim != 1:
            raise ValueError("Input signal must be a 1D array.")
        geometric_mean = scipy.stats.mstats.gmean(np.abs(signal) + 1e-10)
        arithmetic_mean = np.mean(np.abs(signal) + 1e-10)
        spectral_flatness = geometric_mean / arithmetic_mean
    except ValueError as exc:
        logger.error("[ERROR] Failed to calculate spectral flatness: %s", exc)
        spectral_flatness = np.nan

    return spectral_flatness
