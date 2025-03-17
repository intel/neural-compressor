# Called once at the beginning of the test session
def pytest_sessionstart():
    import os
    os.environ.setdefault("EXPERIMENTAL_WEIGHT_SHARING", "FALSE")
