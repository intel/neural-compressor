# Whisper Inference on CPU

## LEGAL DISCLAIMER
To the extent that any data, datasets, or models are referenced by Intel or accessed using tools or code on this site such data, datasets and models are provided by the third party indicated as the source of such content. Intel does not create the data, datasets, or models, provide a license to any third-party data, datasets, or models referenced, and does not warrant their accuracy or quality. By accessing such data, dataset(s) or model(s) you agree to the terms associated with that content and that your use complies with the applicable license. 

Intel expressly disclaims the accuracy, adequacy, or completeness of any data, datasets or models, and is not liable for any errors, omissions, or defects in such content, or for any reliance thereon. Intel also expressly disclaims any warranty of non-infringement with respect to such data, dataset(s), or model(s). Intel is not liable for any liability or damages relating to your use of such data, datasets, or models. 

## Launch the Docker Image
Set the directories on the host system where model, dataset, and log files will reside. These locations will retain model and data content between Docker sessions.
```
export DATA_DIR="${DATA_DIR:-${PWD}/data}"
export MODEL_DIR="${MODEL_DIR:-${PWD}/model}"
export LOG_DIR="${LOG_DIR:-${PWD}/logs}"
```

## Launch the Docker Image
In the Host OS environment, run the following after setting the proper Docker image name. If the Docker image is not on the system already, it will be retrieved from the registry.

If retrieving the model or dataset, ensure any necessary proxy settings are run inside the container.
```
export DOCKER_IMAGE=intel/intel-optimized-pytorch:mlperf-inference-5.1-whisper

docker run --privileged -it --rm \
        --ipc=host --net=host --cap-add=ALL \
        -e http_proxy=${http_proxy} \
        -e https_proxy=${https_proxy} \
        -v ${DATA_DIR}:/data \
        -v ${MODEL_DIR}:/model \
        -v ${LOG_DIR}:/logs \
        --workdir /workspace \
        ${DOCKER_IMAGE} /bin/bash
```

## Prepare workload resources [one-time operations]
Download the model: Run this step inside the Docker container.  This operation will preserve the model on the host system using the volume mapping above.
```
bash scripts/download_model.sh
```
Download the dataset: Run this step inside the Docker container.  This operation will preserve the dataset on the host system using the volume mapping above.
```
bash scripts/download_dataset.sh
```
Calibrate the model: Run this step inside the Docker container.  This operation will create and preserve a calibrated model along with the original model file.
```
bash scripts/run_calibration.sh
```

## Run Benchmark
Run this step inside the Docker container.  Select the appropriate scenario.  If this is the first time running this workload, the original model file will be calibrated to INT8 and stored alongside the original model file (one-time operation).  The default configuration supports Intel EMR.  If running GNR, please make the following [additional changes](GNR.md).

Performance::
```
SCENARIO=Offline MODE=Performance bash run_mlperf.sh
```
Accuracy:
```
SCENARIO=Offline MODE=Accuracy    bash run_mlperf.sh
```

## Run Compliance Tests
Run this step inside the Docker container.  After the benchmark scenarios have been run and results exist in {LOG_DIR}/results, run this step to complete compliance runs. Compliance output will be found in '{LOG_DIR}/compliance'.
```
SCENARIO=Offline MODE=Compliance  bash run_mlperf.sh
```

## Validate Submission Checker
Run this step inside the Docker container.  The following script will perform accuracy log truncation and run the submission checker on the contents of {LOG_DIR}. The source scripts are distributed as MLPerf Inference reference tools. Ensure the submission content has been populated before running.  The script output is transient and destroyed after running.  The original content of ${LOG_DIR} is not modified.
```
VENDOR=Intel bash prepare_submission.sh
```
