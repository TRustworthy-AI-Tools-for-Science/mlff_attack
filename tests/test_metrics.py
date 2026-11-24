import pytest
from mlff_attack import metrics
import numpy as np

def test_calculate_spectral_flatness():
    # Test with a simple signal
    signal = [1, 2, 3, 4, 5]
    sf = metrics.calculate_spectral_flatness(signal)
    assert sf is not None
    assert 0 <= sf <= 1

    # Test with a constant signal
    constant_signal = [5, 5, 5, 5, 5]
    sf_constant = metrics.calculate_spectral_flatness(constant_signal)
    assert sf_constant is not None
    assert np.round(sf_constant, 2) == 1.0

    # Test with an empty signal
    empty_signal = []
    sf_empty = metrics.calculate_spectral_flatness(empty_signal)
    assert np.isnan(sf_empty)

    # Test with a non-1D signal
    non_1d_signal = [[1, 2], [3, 4]]
    sf_non_1d = metrics.calculate_spectral_flatness(non_1d_signal)
    assert np.isnan(sf_non_1d)
