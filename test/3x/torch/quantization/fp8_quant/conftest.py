# Called once at the beginning of the test session
def pytest_sessionstart():
    import os

    os.environ.setdefault("PT_HPU_LAZY_MODE", "1")
    os.environ.setdefault("EXPERIMENTAL_WEIGHT_SHARING", "FALSE")
