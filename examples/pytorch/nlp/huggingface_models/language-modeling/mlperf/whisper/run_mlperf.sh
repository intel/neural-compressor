#!/bin/bash

set -x

# Controls workload mode
export SCENARIO="${SCENARIO:-Offline}"
export MODE="${MODE:-Performance}"
export OFFLINE_QPS="${OFFLINE_QPS:-0}"
export SERVER_QPS="${SERVER_QPS:-0}"
export AUTO_USER_CONF="${AUTO_USER_CONF:-True}"
export SYSTEM="${SYSTEM:-AUTO}"
export DEBUG="${DEBUG:-False}"

# Setting standard environmental paths
export WORKSPACE_DIR=/workspace
export DATA_DIR=/data
export MODEL_DIR=/model
export LOG_DIR=/logs
export DOCUMENTATION_DIR=${LOG_DIR}/documentation
export CODE_DIR=${LOG_DIR}/code
export COMPLIANCE_DIR=${LOG_DIR}/compliance
export MEASUREMENTS_DIR=${LOG_DIR}/measurements
export RESULTS_DIR=${LOG_DIR}/results
export SYSTEMS_DIR=${LOG_DIR}/systems

##########     SUPPORT FUNCTIONS BEGIN HERE     ##########

# Set HW specific qps settings from a select list of SKUs, or default.
configure_system () {
  export SYSTEM="1-node-1x-BMG-Pro-B60-Dual"
#  export NUM_CORES=`lscpu -b -p=Core,Socket | grep -v '^#' | sort -u | wc -l`
#  if [ "${SYSTEM}" != "AUTO" ] ; then
#      export SYSTEM="${SYSTEM}"
#  elif [ "${NUM_CORES}" == "256" ]; then
#      export SYSTEM="1-node-2S-GNR_128C"
#  elif [ "${NUM_CORES}" == "192" ]; then
#      export SYSTEM="1-node-2S-GNR_96C"
#  elif [ "${NUM_CORES}" == "172" ]; then
#      export SYSTEM="1-node-2S-GNR_86C"
#  else
#      export SYSTEM="DEFAULT"
#  fi

  echo ${SYSTEM}
}

# Creates the default user.conf file, either auto-selected, modified, or newly generated.
configure_userconf () {
  # cd ${WORKSPACE_DIR}

  # Ensure no left-over user.conf files from previous runs, and use pre-configured SYSTEM file if available.
  if [ -f "${USER_CONF}" ]; then rm ${USER_CONF}; fi
  if [ -f "systems/user.conf.${SYSTEM}" ]; then cp systems/user.conf.${SYSTEM} ${USER_CONF}; fi

  # If an Offline QPS is manually specified, modify the existing user.conf or add to a new one.
  if [ "${OFFLINE_QPS}" != "0" ]; then
      if [ "$(grep "Offline.target_qps" user.conf | wc -l)" == "0" ]; then
          echo "*.Offline.target_qps = ${OFFLINE_QPS}" >> ${USER_CONF}
      else
          sed -i 's/.*Offline.target_qps.*/\*\.Offline\.target_qps = '"${OFFLINE_QPS}"'/g' ${USER_CONF}
      fi
  fi

  # If a Server QPS is manually specified, modify the existing user.conf or add to a new one.
  if [ "${SERVER_QPS}" != "0" ]; then
      if [ "$(grep "Server.target_qps" user.conf | wc -l)" == "0" ]; then
          echo "*.Server.target_qps = ${SERVER_QPS}" >> ${USER_CONF}
      else
          sed -i 's/.*Server.target_qps.*/\*\.Server\.target_qps = '"${SERVER_QPS}"'/g' ${USER_CONF}
      fi
  fi
}

# Creates the non-run-specific submission files (necessary for final submission).
prepare_suplements () {
  # cd ${WORKSPACE_DIR}
  # Ensure /logs/systems is populated or abort process.
  if [ -f "systems/${SYSTEM}.json" ]; then
    cp systems/${SYSTEM}.json ${SYSTEMS_DIR}/
  else
    echo '{ "submitter": "OEM", "system_name": "DEFAULT" }' > ${SYSTEMS_DIR}/${SYSTEM}.json
  fi

  # Populate /logs/code directory
  cp -r README.md ${CODE_PATH}/

  # Populate /logs/measurements directory
  cp measurements.json ${MEASUREMENTS_PATH}/${SYSTEM}.json
  cp README.md user.conf scripts/run_calibration.sh ${MEASUREMENTS_PATH}/

  # Populate /logs/documentation directory
  cp calibration.md ${DOCUMENTATION_DIR}/
}

# Initializes the system for an MLPerf run, then launches the run.
run_workload () {
  # cd ${WORKSPACE_DIR}
  if [ "${DEBUG}" == "False" ] ; then bash run_clean.sh; fi
  if [ -f "${RUN_LOGS}" ]; then rm -r ${RUN_LOGS}; fi
  mkdir -p ${RUN_LOGS}
  workload_specific_run
}

# Places the standard MLPerf run log outputs to the specified final dir.
stage_logs () {
  OUTPUT_PATH=$1
  cd ${RUN_LOGS}
  mkdir -p ${OUTPUT_PATH}
  mv mlperf_log_accuracy.json mlperf_log_detail.txt mlperf_log_summary.txt ${OUTPUT_PATH}/
  if [ -f accuracy.txt ]; then mv accuracy.txt ${OUTPUT_PATH}/; fi
}

##########     RUN BEGINS HERE     ##########

# Using workload-specific parameters from 'configure_workload.sh', create the submission dir structure.
source configure_workload.sh
export SYSTEM="$(configure_system)"
export CODE_PATH=${CODE_DIR}/${WORKLOAD}/${IMPL}
export MEASUREMENTS_PATH=${MEASUREMENTS_DIR}/${SYSTEM}/${WORKLOAD}/${SCENARIO}
export COMPLIANCE_PATH=${COMPLIANCE_DIR}/${SYSTEM}/${WORKLOAD}/${SCENARIO}
export RESULTS_PATH=${RESULTS_DIR}/${SYSTEM}/${WORKLOAD}/${SCENARIO}
mkdir -p ${SYSTEMS_DIR}
mkdir -p ${CODE_PATH}
mkdir -p ${MEASUREMENTS_PATH}
mkdir -p ${DOCUMENTATION_DIR}

# Ensuring the user.conf file is created if auto is enabled. If disabled, checks for existing one.
export USER_CONF=user.conf
if [ "${AUTO_USER_CONF}" == "True" ]; then configure_userconf; fi
if [ -f "${USER_CONF}" ]; then
  echo "LOG:::: Contents of user.conf:"
  cat ${USER_CONF}
else
  echo "ERROR::: No user.conf file found."
fi

# Creates the non-runtime submission content (code, systems, measurements)
if [ "${DEBUG}" == "False" ] ; then prepare_suplements; fi

# Beginning workload runs, with Mode of: Performance, Accuracy, OR Compliance
export RUN_LOGS=${WORKSPACE_DIR}/run_output
if [ "${MODE}" == "Performance" ]; then
    run_workload
    stage_logs "${RESULTS_PATH}/performance/run_1"
elif [ "${MODE}" == "Accuracy" ]; then
    run_workload
    stage_logs "${RESULTS_PATH}/accuracy"
elif [ "${MODE}" == "Compliance" ]; then
    for TEST in ${COMPLIANCE_TESTS}; do
        echo "Running compliance ${TEST} ..."
        
	if [ -f ${WORKSPACE_DIR}/audit.config ]; then rm ${WORKSPACE_DIR}/audit.config; fi
        if [ "$TEST" == "TEST01" ]; then
            cp ${COMPLIANCE_SUITE_DIR}/${TEST}/${MODEL}/audit.config .
        else
            cp ${COMPLIANCE_SUITE_DIR}/${TEST}/audit.config .
        fi

	if ! [ -d ${RESULTS_PATH} ]; then
	    echo "[ERROR] Compliance run could not be verified due to unspecified or non-existent RESULTS_PATH: ${RESULTS_PATH}"
            exit
	fi

        OUTPUT_PATH=${RUN_LOGS}
	run_workload

        python ${COMPLIANCE_SUITE_DIR}/${TEST}/run_verification.py -r ${RESULTS_PATH} -c ${OUTPUT_PATH} -o ${COMPLIANCE_PATH}
    done
else
    echo "[ERROR] Missing value for MODE. Options: Performance, Accuracy, Compliance"
fi
