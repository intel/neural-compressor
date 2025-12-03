# Copyright (c) 2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
#
# THIS IS A GENERATED DOCKERFILE.
#
# This file was assembled from multiple pieces, whose use is documented
# throughout. Please refer to the TensorFlow dockerfiles documentation
# for more information.

ARG BASE_IMAGE=intel/deep-learning-essentials
ARG BASE_TAG=2025.1.3-0-devel-ubuntu24.04
FROM ${BASE_IMAGE}:${BASE_TAG} AS dev-base

#Removing private keys from BASE
RUN rm /etc/ssh/ssh_host_*_key && \
    rm /etc/ssh/ssh_host_*_key.pub

ARG http_proxy
ARG https_proxy
#ARG no_proxy

ENV http_proxy=${http_proxy}
ENV https_proxy=${https_proxy}
#ENV no_proxy=${no_proxy}

COPY ./ /workspace/
WORKDIR /workspace

ENV DEBIAN_FRONTEND=noninteractive

RUN wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor | tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null && \
    echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | tee /etc/apt/sources.list.d/oneAPI.list && \
    add-apt-repository -y ppa:kobuk-team/intel-graphics-testing

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get autoclean && \
    apt-get autoremove -y && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get install -y --no-install-recommends \
                    gcc \
                    g++ \
                    python3.12 \
                    python3.12-venv \
                    python3.12-dev \
		    python3-pip \
                    python-is-python3 \
                    vim \
                    wget \
                    rsync \
                    git \
                    bc \
                    unzip \
                    zlib1g-dev \
                    libgl1 # && \
    # curl -skS https://bootstrap.pypa.io/get-pip.py | python3.10

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 

# oneccl
# TODO: Add script to download oneccl
RUN cd /workspace && \
    wget https://github.com/uxlfoundation/oneCCL/releases/download/2021.15.4/intel-oneccl-2021.15.4.11_offline.sh && \
    bash /workspace/intel-oneccl-2021.15.4.11_offline.sh -a --silent --eula accept && echo "source /opt/intel/oneapi/setvars.sh --force" >> /root/.bashrc && \
    rm -rf /workspace/intel-oneccl-2021.15.4.11_offline.sh

RUN apt install -y --allow-downgrades \
                   libze-intel-gpu1=25.18.33578.15-1146~24.04 \
                   libze1=1.21.9.0-1136~24.04 \
                   libze-dev=1.21.9.0-1136~24.04 \
                   intel-opencl-icd=25.18.33578.15-1146~24.04 \
                   intel-ocloc=25.18.33578.15-1146~24.04 \
                   libigsc0 intel-gsc libmetee4 && \
    wget https://github.com/intel/xpumanager/releases/download/V1.3.0/xpu-smi_1.3.0_20250707.103634.3db7de07.u24.04_amd64.deb && \
    dpkg -i xpu-smi*.deb && \
    rm -rf /var/lib/apt/lists/* && \
    echo "alias ll='ls -l'" >> ~/.bashrc

RUN python -m pip config set global.break-system-packages true && \
    pip install torch==2.8.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu && \
    pip install packaging librosa openai-whisper ftfy

# RUN pip install --no-deps accelerate==0.29.3 threadpoolctl==3.6.0 git+https://github.com/intel/auto-round.git@mlperf-awq

# Bypass PEP 668
RUN python -m pip config set global.break-system-packages true && \
    cd /workspace/ && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-deps dist/*.whl # && rm -rf dist && bash prepare_loadgen.sh

RUN mkdir -p /workspace/third_party && \
    cd /workspace/third_party && \
    git clone https://github.com/mlcommons/inference.git mlperf-inference && \
    cd mlperf-inference && \
    git checkout b9ed3c7 && \
    cd loadgen && \
    python3 -m pip install . && \
    cp ../mlperf.conf /workspace/

# Cleanup internal directories
RUN for FILE in $(cat /workspace/redactions.txt); do rm -rf /workspace/${FILE}; done
