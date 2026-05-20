def pytest_configure(config):
    config.addinivalue_line(
        "markers", "unit_test: marks test as a unit test, which takes at most a couple of seconds to finish"
    )
    config.addinivalue_line(
        "markers",
        "model_test(model): states the test involves quantization of a big keras model, so they take much more time to finish",
    )
