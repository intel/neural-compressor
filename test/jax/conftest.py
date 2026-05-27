import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "CI_test: marks test to be run in CI")
    config.addinivalue_line(
        "markers", "CI_test_if: marks test to be run in CI if combination of parameters matches arguments"
    )


def pytest_collection_modifyitems(items):
    for item in items:
        marker_CI_test_if = item.get_closest_marker("CI_test_if")
        if marker_CI_test_if is not None:
            all_test_parameters = item.nodeid[item.nodeid.rfind("[") + 1 : -1]
            parameter_sets_to_be_marked = (
                marker_CI_test_if.args if isinstance(marker_CI_test_if.args[0], list) else [marker_CI_test_if.args]
            )

            # If parameters in test case match all arguments passed to CI_test_if marker, marks them with CI_test
            for parameter_set in parameter_sets_to_be_marked:
                if all(param in all_test_parameters for param in parameter_set):
                    item.add_marker(pytest.mark.CI_test)
