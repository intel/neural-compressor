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

set -x

echo "[INFO] Start installing software packages and dependencies"

# install machine packages
sudo apt-get -y update
sudo apt-get install -y wget 
sudo apt-get install -y git
sudo apt-get install -y build-essential
sudo apt-get install -y htop aha html2text numactl bc
sudo apt-get install -y ffmpeg libsm6 libxext6 
sudo apt-get install -y automake libtool
sudo apt-get install -y python3 pip

# install conda
wget https://repo.continuum.io/archive/Anaconda3-5.0.0-Linux-x86_64.sh -O anaconda3.sh
chmod +x anaconda3.sh
sudo ./anaconda3.sh -b -p /home/anaconda3
export PATH=/home/anaconda3/bin:$PATH
conda create -yn test python=3.9
source activate test

# install pip modules
pip install numpy
pip install pyyaml
pip install typing_extensions
pip install psutil
pip install neural_compressor intel_extension_for_pytorch 

# install torch
pip3 install torch torchvision torchaudio
pip3 install torchdynamo
pip3 install transformers
