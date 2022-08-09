#
# Copyright (c) 2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

ARG UBUNTU_VER=20.04
FROM ubuntu:${UBUNTU_VER} as deploy

# See http://bugs.python.org/issue19846
ENV LANG C.UTF-8
ARG PYTHON=python3.8

RUN apt-get update && apt-get install -y --no-install-recommends --fix-missing \
    ${PYTHON}-dev \
    gcc \
    libgl1-mesa-glx \
    libglib2.0-0 \
    python3 \
    python3-pip

RUN ${PYTHON} -m pip --no-cache-dir install --upgrade \
    pip \
    setuptools

RUN ln -sf $(which ${PYTHON}) /usr/local/bin/python && \
    ln -sf $(which ${PYTHON}) /usr/local/bin/python3 && \
    ln -sf $(which ${PYTHON}) /usr/bin/python && \
    ln -sf $(which ${PYTHON}) /usr/bin/python3

ARG INC_VER=1.12

ADD https://raw.githubusercontent.com/intel/neural-compressor/v${INC_VER}/third-party-programs.txt /licenses/
ADD https://raw.githubusercontent.com/intel/neural-compressor/v${INC_VER}/docker/third-party-programs-docker.txt /licenses/
ADD https://raw.githubusercontent.com/intel/neural-compressor/v${INC_VER}/LICENSE /licenses/

RUN python -m pip install --no-cache-dir neural-compressor${INC_VER:+==${INC_VER}}
