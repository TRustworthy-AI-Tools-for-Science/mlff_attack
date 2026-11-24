"""Tests for mlff_attack package."""
import pytest
from mlff_attack import __version__

@pytest.mark.smoke
def test_version():
    assert __version__ == "0.1.0"

@pytest.mark.smoke
def test_package_import():
    import mlff_attack
    assert mlff_attack is not None
