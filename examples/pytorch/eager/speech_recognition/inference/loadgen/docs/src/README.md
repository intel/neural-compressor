# Generating the HTML docs {#ReadmeHtmlDocs}

This document is generated from inline docstrings in the source and
various markdown files checked into the git repository. If you've
checked out the code, you can generate this documentation.

*Prerequisite:* You must have [doxygen](http://www.doxygen.nl) installed
on your system:

## With gn / ninja

If you are using the gn build flow, you may run:

    ninja -C out/Release generate_doxygen_html

* This will output the documentation to out/Release/gen/loadgen/docs/gen and
avoid poluting the source directory.

## Manually

Alternatively, you can manually run:

    python docs/src/doxygen_html_generator.py <target_dir> <loadgen_root>

* If <loadgen_root> is omitted, it will default to ".".
* If <target_dir> is also omitted, it will default to "./docs/gen".

## Hosting

A version of this doc is currently hosted online at
https://mlperf.github.io/inference/loadgen/index.html

To update the hosted version, submit a PR to the
[mlperf.github.io](https://github.com/mlperf/mlperf.github.io) repository.
