# Building the LoadGen {#ReadmeBuild}

## Prerequisites

    sudo apt-get install libglib2.0-dev python-pip python3-pip
    pip2 install absl-py numpy
    pip3 install absl-py numpy

## Quick Start

    pip install absl-py numpy
    git clone --recurse-submodules https://github.com/mlperf/inference.git mlperf_inference
    cd mlperf_inference/loadgen
    CFLAGS="-std=c++14 -O3" python setup.py bdist_wheel
    pip install --force-reinstall dist/mlperf_loadgen-0.5a0-cp36-cp36m-linux_x86_64.whl
    python demos/py_demo_single_stream.py

This will fetch the loadgen source, build and install the loadgen as a python module, and run a simple end-to-end demo. The exact *.whl filename may differ on your system, but there should only be one resulting whl file for you to use.

A summary of the test results can be found in the *"mlperf_log_summary.txt"* logfile.

For a timeline visualization of what happened during the test, open the *"mlperf_log_trace.json"* file in Chrome:
* Type “chrome://tracing” in the address bar, then drag-n-drop the json.
* This may be useful for SUT performance tuning and understanding + debugging the loadgen.

To build the loadgen as a C++ library, rather than a python module:

    git clone --recurse-submodules https://github.com/mlperf/inference.git mlperf_inference
    cd mlperf_inference
    make mlperf_loadgen
    cp out/MakefileGnProj/obj/loadgen/libmlperf_loadgen.a .

Alternatively:

    git clone https://github.com/mlperf/inference.git mlperf_inference
    cd mlperf_inference
    mkdir loadgen/build/ && cd loadgen/build/
    cmake .. && cmake --build .
    cp libmlperf_loadgen.a ..

## Overview

The load generator is built using the
[gn metabuild](https://gn.googlesource.com/gn/+/master)
and [ninja build](https://ninja-build.org/) tools.

Using git submodules to manage a checkout will compile gn and ninja if needed.
Using depot\_tools to manage a checkout will include prebuilt versions of gn and ninja.
By default, both ways will use the system C++ compiler, however depot\_tools also
has the option to use versioned compiler binaries.

If the tools above don't cover your particular configuration, please reach out.
Patches to support other build environments welcome.

## Git Submodules Approach

### Downloading the source

Download the mlperf inference repository and its submodules.

    git clone --recurse-submodules https://github.com/mlperf/inference.git

### Build from source

#### Just building once

The following is a phony Makefile target, wrapping everything needed to build
the load generator. It'll get the job done, but will result in lots of redundant
work if used over and over again, so isn't recommended for development.
The resulting binary will be found in: out/MakefileGnProj/obj/loadgen/libmlperf\_loadgen.*

    make mlperf_loadgen

To build the python module:

    make mlperf_loadgen_pymodule

Note: These make targets assume Unix-like shell commands are available.

#### Building for development purposes

The bootstrap\_gn\_ninja make target will build ninja and gn into the current
tree and should only need to be called once per checkout.
This is not necessary if you already have gn and ninja installed on your
system.

    make bootstrap_gn_ninja

Then generate the ninja build files from the gn build files.
You can have multiple sets of ninja build files in different out directories,
each with their own args. Release and debug sets, for example:

    third_party/gn/gn gen out/Release --args="is_debug=false"
    third_party/gn/gn gen out/Debug --args="is_debug=true"

From here you can edit+build over and over using one of the following targets:

    third_party/ninja/ninja -C out/Release mlperf_loadgen
    third_party/ninja/ninja -C out/Release loadgen_pymodule_wheel_src

You will find the binary in *"out/Release/obj/loadgen/libmlperf\_loadgen.a"*. (Or as a .lib on Windows).

Link that library directly into the executable you want to test.

Optionally, create a project file for your favorite IDE.
See [gn documentation](https://gn.googlesource.com/gn/+/master/docs/reference.md#ide-options) for details.

    gn gen --root=src --ide=eclipse out/Release
    gn gen --root=src --ide=vs out/Release
    gn gen --root=src --ide=xcode out/Release
    gn gen --root=src --ide=qtcreator out/Release

## Depot Tools approach

### Downloading the source

Download and install depot\_tools:

    git clone 'https://chromium.googlesource.com/chromium/tools/depot_tools.git'
    export PATH="${PWD}/depot_tools:${PATH}"

Copy the depot\_tools fetch config for the MLPerf load generator to the
depot\_tools path:

    wget https://raw.githubusercontent.com/mlperf/inference/master/loadgen/depot_tools/fetch_configs/mlperf_loadgen.py \
    -O ${PWD}/depot_tools/fetch_configs/mlperf_loadgen.py

Create a folder for the load generator project and fetch the source code:

    mkdir mlperf_loadgen
    cd mlperf_loadgen
    fetch mlperf_loadgen
    gclient sync

### Building from source

The depot\_tools approach enables building with versioned toolchains.

<i>Note: This has only been tested to work in a Debian-based linux environment so
far, but it should support many others.</i>

Run the gn metabuild. The output directory will contain ninja build files for a
specific target platform and set of build options.

    gn gen --root=src out/Default

TODO: Provide gn commands for cross-compiling to Android and to iOS. In the mean
time, looking at how to build chromium
[for android](https://chromium.googlesource.com/chromium/src/+/master/docs/android_build_instructions.md)
or [for ios](https://chromium.googlesource.com/chromium/src/+/HEAD/docs/ios/build_instructions.md)
should provide some useful pointers.

Build the library:

    ninja -C out/Default mlperf_loadgen

You will find the binary in *"out/Release/obj/loadgen/libmlperf\_loadgen.a"*. (Or as a .lib on Windows).

Link that library directly into the executable you want to test.

## Notes about important dependency and build files

**DEPS**: Describes source repositories to pull in as dependencies depot\_tools'
"gclient sync" command. Also runs system commands to download the relevant toolchains
for the "gclient runhooks" command, which is run as part of the initial "fetch".

**.gitmodules**: Git submodules alternative to DEPS.

**.gn**: The root gn build file.

**BUILD.gn**: Located in multiple directories. Each one describes how to build
that particular directory.
