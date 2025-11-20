# conftest.py
def pytest_configure(config):
    config.addinivalue_line(
        "markers", "smoke: mark as smoke test", "cli: make as cli test"
    )