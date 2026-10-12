# Copyright (c) 2025 Intel Corporation
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

#!/bin/bash

export WORKSPACE_DIR=/workspace
export DATA_DIR=/data
export LIBRISPEECH_DIR=${DATA_DIR}/LibriSpeech
export UTILS_DIR=${WORKSPACE_DIR}/utils
mkdir -p ${LIBRISPEECH_DIR}

cd ${WORKSPACE_DIR}

# Downloads all Librispeech dev partitions
python ${UTILS_DIR}/download_librispeech.py \
    ${UTILS_DIR}/inference_librispeech.csv \
    ${LIBRISPEECH_DIR} \
    -e ${DATA_DIR}

# Consolidates all Librispeech partitions into common dir
mkdir -p ${LIBRISPEECH_DIR}/dev-all
cp -r ${LIBRISPEECH_DIR}/dev-clean/* \
      ${LIBRISPEECH_DIR}/dev-other/* \
      ${LIBRISPEECH_DIR}/dev-all/

# Coverts original Librispeech flac to wav and creates manifest file
python ${UTILS_DIR}/convert_librispeech.py \
   --input_dir ${LIBRISPEECH_DIR}/dev-all \
   --dest_dir ${DATA_DIR}/dev-all \
   --output_json ${DATA_DIR}/dev-all.json

# Repackages Librispeech samples into samples approaching 30s
python utils/repackage_librispeech.py --manifest ${DATA_DIR}/dev-all.json \
	                              --data_dir ${DATA_DIR} \
				      --output_dir ${DATA_DIR}/dev-all-repack \
				      --output_json /data/dev-all-repack.json
