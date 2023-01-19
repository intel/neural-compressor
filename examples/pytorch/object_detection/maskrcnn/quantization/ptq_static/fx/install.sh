#!/bin/bash

# Installs object_detection module

pushd pytorch

rm -Rf build/
python setup.py clean build develop --user

popd
