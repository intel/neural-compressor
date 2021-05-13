# Building and Running the Tests {#ReadmeTests}

The unit and performance tests are only supported via gn/ninja at the moment.

See the [top-level build readme](@ref ReadmeBuild) for details but, from a clean checkout, you must first run:

    make bootstrap_gn_ninja
    third_party/gn/gn gen out/Release --args="is_debug=false"

This will build the gn and ninja build tools and create a release project.

## Unit Tests

To build:

    third_party/ninja/ninja -C out/Release mlperf_loadgen_tests_basic

To run all tests:

    out/Release/mlperf_loadgen_tests_basic .

To run specific tests:

    out/Release/mlperf_loadgen_tests_basic <regex>
    e.g.:
    out/Release/mlperf_loadgen_tests_basic SingleStream

## Performance Tests

To build:

    third_party/ninja/ninja -C out/Release mlperf_loadgen_perftests

To run all tests:

    out/Release/mlperf_loadgen_perftests .

To run specific tests:

    out/Release/mlperf_loadgen_perftests <regex>
    e.g.:
    out/Release/mlperf_loadgen_tests_basic ServerPool
