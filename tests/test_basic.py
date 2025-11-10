"""Tests for mlff_attack package."""
import pytest
from mlff_attack import __version__


def test_version():
    """Test that version is defined."""
    assert __version__ == "0.1.0"


def test_import():
    """Test that package can be imported."""
    import mlff_attack
    assert mlff_attack is not None
