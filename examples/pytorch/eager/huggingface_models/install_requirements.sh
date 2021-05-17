#!/bin/bash

function main {
  init_params "$@"
  install_requirements
}

# Initialize parameters
function init_params {
  for var in "$@"
  do
    case $var in
      --topology=*)
          topology=$(echo $var |cut -f2 -d=)
      ;;
      *)
          echo "Error: No such parameter: ${var}"
          exit 1
      ;;
    esac
  done
}

# Install model requirements
function install_requirements {
    SCRIPT_DIR=examples/text-classification
    if [ "${topology}" = "t5_WMT_en_ro" ];then
        SCRIPT_DIR="examples/seq2seq"
    elif [ "${topology}" = "marianmt_WMT_en_ro" ]; then
        SCRIPT_DIR="examples/seq2seq"
    elif [ "${topology}" = "pegasus_billsum" ]; then
        SCRIPT_DIR="examples/seq2seq"
    elif [ "${topology}" = "dialogpt_wikitext" ]; then
        SCRIPT_DIR="examples/language-modeling"
    elif [ "${topology}" = "reformer_crime_and_punishment" ]; then
        SCRIPT_DIR="examples/language-modeling"
    fi

    pushd ${SCRIPT_DIR}
    python -m pip install -r requirements.txt
    popd
}
main "$@"
