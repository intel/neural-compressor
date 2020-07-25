#!/bin/bash
# set -x

DATA_DIR="./data"
DATA_NAME="val_256_q90.rec"
DATA_URL="http://data.mxnet.io/data/val_256_q90.rec"

help()
{
   cat <<- EOF

   Desc: Prepare dataset for MXNet ImageNet Classfication.

   -h --help              help info

   --dataset_location     set dataset location, default is ./data

EOF
   exit 0
}

function main {

  init_params "$@"
  download_dataset

}

# init params
function init_params {

  for var in "$@"
  do
    case $var in
      --dataset_location=*)
          DATA_DIR=$(echo $var |cut -f2 -d=)
      ;;
      -h|--help) help
      ;;
      *)
      echo "Error: No such parameter: ${var}"
      exit 1
      ;;
    esac
  done

}

# download_dataset
function download_dataset {

  if [ ! -d ${DATA_DIR} ]; then
    mkdir ${DATA_DIR}
  fi

  cd ${DATA_DIR}
  if [ ! -f ${DATA_NAME} ]; then
    wget http://data.mxnet.io/data/val_256_q90.rec
  else
    echo "Dataset ${DATA_NAME} is exist!"
  fi

  cd ../

}

main "$@"