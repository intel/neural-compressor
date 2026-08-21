#!/bin/bash

export LOG_DIR=/logs
export SUBMISSION_DIR=${LOG_DIR}/submission-$(date +%s)
export SUBMISSION_ORIGINAL=${SUBMISSION_DIR}/original
export SUBMISSION_PROCESSED=${SUBMISSION_DIR}/processed

RESULTS_DIR=${LOG_DIR}/results
COMPLIANCE_DIR=${LOG_DIR}/compliance
MEASUREMENTS_DIR=${LOG_DIR}/measurements
SYSTEMS_DIR=${LOG_DIR}/systems
echo "Ensuring correct system directories and files match system ${SYSTEM}."
echo "The following are expected:"
echo "- RESULTS: ${RESULTS_DIR}/${SYSTEM}"
echo "- COMPLIANCE: ${COMPLIANCE_DIR}/${SYSTEM}"
echo "- MEASUREMENTS: ${MEASUREMENTS_DIR}/${SYSTEM}"
echo "- SYSTEM FILE: ${SYSTEMS_DIR}/${SYSTEM}.json"
if ! [ -d ${RESULTS_DIR}/${SYSTEM} ];      then echo "[ERROR] RESULTS_DIR not found: ${RESULTS_DIR}/${SYSTEM}";           exit; fi
if ! [ -d ${COMPLIANCE_DIR}/${SYSTEM} ];   then echo "[ERROR] COMPLIANCE_DIR not found: ${COMPLIANCE_DIR}/${SYSTEM}";     exit; fi
if ! [ -d ${MEASUREMENTS_DIR}/${SYSTEM} ]; then echo "[ERROR] MEASUREMENTS_DIR not found: ${MEASUREMENTS_DIR}/${SYSTEM}"; exit; fi
if ! [ -f ${SYSTEMS_DIR}/${SYSTEM}.json ]; then echo "[ERROR] SYSTEM file not found: ${SYSTEMS_DIR}/${SYSTEM}.json";      exit; fi

echo "Verifying correct 'submitter' and 'system_name' fields in: ${SYSTEMS_DIR}/${SYSTEM}.json. These should match 'config/workload.conf'."
if ! (( $(grep -r "\"submitter\": \"${VENDOR}\"" ${SYSTEMS_DIR}/${SYSTEM}.json | wc -l) > 0 ));   then echo "[ERROR] Field 'submitter' does not match 'VENDOR'.";   exit; fi
if ! (( $(grep -r "\"system_name\": \"${SYSTEM}\"" ${SYSTEMS_DIR}/${SYSTEM}.json | wc -l) > 0 )); then echo "[ERROR] Field 'system_name' does not match 'SYSTEM'."; exit; fi

mkdir -p ${SUBMISSION_ORIGINAL}/closed/${VENDOR}
cp -r ${LOG_DIR}/code \
      ${LOG_DIR}/compliance \
      ${LOG_DIR}/documentation \
      ${LOG_DIR}/measurements \
      ${LOG_DIR}/results \
      ${LOG_DIR}/systems \
      ${SUBMISSION_ORIGINAL}/closed/${VENDOR}/

echo "Truncating the logs: ${SUBMISSION_ORIGINAL} --> ${SUBMISSION_PROCESSED}"
cd /workspace/third_party
python mlperf-inference/tools/submission/truncate_accuracy_log.py --input ${SUBMISSION_ORIGINAL} --submitter ${VENDOR} --output ${SUBMISSION_PROCESSED}

echo "Running submission checker: ${SUBMISSION_PROCESSED}"
cd /workspace/third_party
python3 mlperf-inference/tools/submission/submission_checker.py --input ${SUBMISSION_PROCESSED} --submitter=${VENDOR} --version=v5.1
