#!/bin/bash
# set -x

DATA_DIR="${PWD}/data"
DATA_NAME="val2017"
DATA_URL_LIST='http://images.cocodataset.org/zips/val2017.zip http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
PACKAGES_LIST='val2017.zip annotations_trainval2017.zip'
VAL_IMAGE_DIR=$DATA_DIR/val2017
TRAIN_ANNOTATIONS_FILE=$DATA_DIR/annotations/empty.json
VAL_ANNOTATIONS_FILE=$DATA_DIR/annotations/instances_val2017.json
TESTDEV_ANNOTATIONS_FILE=$DATA_DIR/annotations/empty.json
OUTPUT_DIR=$DATA_DIR

help()
{
   cat <<- EOF

   Desc: Prepare dataset for Tensorflow COCO object detection.

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
          DATA_DIR=$(echo "$var" |cut -f2 -d=)
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

# removes files that will not be used anymore
function remove_zipped_packages {
  for package in $PACKAGES_LIST; do
    rm "$package"
  done
}

# download_dataset
function download_dataset {
  if [ ! -d "${DATA_DIR}" ]; then
    mkdir "${DATA_DIR}"
  fi

  cd "${DATA_DIR}" || exit
  if [ ! -f "${VAL_IMAGE_DIR}" ]; then

    for dataset_dowload_link in $DATA_URL_LIST; do
      wget "$dataset_dowload_link"
    done
    for package in $PACKAGES_LIST; do
      unzip -o "$package"
    done
    remove_zipped_packages
    if [ ! -d empty_dir ]; then
      mkdir empty_dir
    fi

    cd annotations || exit
    echo "{ \"images\": {}, \"categories\": {}}" > empty.json
    cd ..
  else
    echo "Dataset ${DATA_NAME} is exist!"
  fi

  cd ../
}

main "$@"
