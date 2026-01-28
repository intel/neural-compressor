#!/bin/bash

export IMAGE_NAME="tiyengar:base_xpu"

echo "Building XPU MLPerf workflow container"

# Build the container
docker build \
    -f docker/Dockerfile \
    --build-arg http_proxy=${http_proxy} \
    --build-arg https_proxy=${https_proxy} \
    -t ${IMAGE_NAME} .
