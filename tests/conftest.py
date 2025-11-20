# conftest.py
def pytest_configure(config):
    config.addinivalue_line(
        "markers", "smoke: mark as smoke test"
    )
    config.addinivalue_line(
        "markers", "cli: mark as cli test"
    )