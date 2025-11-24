# MLPerf loadgen
mkdir -p /workspace/third_party && cd /workspace/third_party && rm -rf mlperf-inference && \
git clone https://github.com/mlcommons/inference.git mlperf-inference && cd mlperf-inference && git checkout b9ed3c7 && cd loadgen && \
python3 -m pip install . && cp ../mlperf.conf ../../../ && cd ../../../
