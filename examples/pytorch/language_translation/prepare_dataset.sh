#!/bin/bash
set -x

function main {

  init_params "$@"
  run_tuning

}

# init params
function init_params {

  for var in "$@"
  do
    case $var in
      --dataset_location=*)
          dataset_location=$(echo $var |cut -f2 -d=)
      ;;
      *)
          echo "Error: No such parameter: ${var}"
          exit 1
      ;;
    esac
  done

}

# run_tuning
function run_tuning {
    wget -P ${dataset_location} https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e/raw/17b8dd0d724281ed7c3b2aeeda662b92809aadd5/download_glue_data.py
    chmod a+x ${dataset_location}/download_glue_data.py
    python ${dataset_location}/download_glue_data.py --data_dir ${dataset_location} --tasks RTE,MRPC,CoLA,STS,QNLI,SST
    rm ${dataset_location}/download_glue_data.py
    mkdir -p ${dataset_location}/SQuAD
    wget -P ${dataset_location}/SQuAD https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json
    wget -P ${dataset_location}/SQuAD https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json
    wget -P ${dataset_location}/SQuAD https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json
    wget -P ${dataset_location}/SQuAD https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json

}

main "$@"
