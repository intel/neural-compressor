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
  convert_to_tf_record
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

function download_tf_models_repo {
  if [ ! -d models ]; then
    git clone https://github.com/tensorflow/models.git
  fi
  cd models || exit
  git checkout 7a9934df2afdf95be9405b4e9f1f2480d748dc40
  cd ..
}

function divide_tf_records_by_dataset {
  if [ ! -d "${DATA_DIR}/tf_test2017" ]; then
    mkdir "${DATA_DIR}/tf_test2017"
  fi
  if [ ! -d "${DATA_DIR}/tf_train2017" ]; then
    mkdir "${DATA_DIR}/tf_train2017"
  fi
  if [ ! -d "${DATA_DIR}/tf_val2017" ]; then
    mkdir "${DATA_DIR}/tf_val2017"
  fi
  mv ${DATA_DIR}/coco_testdev.record* ${DATA_DIR}/tf_test2017
  mv ${DATA_DIR}/coco_train.record* ${DATA_DIR}/tf_train2017
  mv ${DATA_DIR}/coco_val.record* ${DATA_DIR}/tf_val2017
}

function convert {
  cd models/research
  protoc object_detection/protos/*.proto --python_out=.
  export PYTHONPATH=$PYTHONPATH:$(pwd)
  export PYTHONPATH=$PYTHONPATH:$(pwd)/slim
  python ./object_detection/dataset_tools/create_coco_tf_record.py --logtostderr \
    --train_image_dir=empty_dir \
    --val_image_dir="${VAL_IMAGE_DIR}" \
    --test_image_dir=empty_dir \
    --train_annotations_file="${TRAIN_ANNOTATIONS_FILE}" \
    --val_annotations_file="${VAL_ANNOTATIONS_FILE}" \
    --testdev_annotations_file="${TESTDEV_ANNOTATIONS_FILE}" \
    --output_dir="${OUTPUT_DIR}"
}

function convert_to_tf_record {
  download_tf_models_repo
  convert
  divide_tf_records_by_dataset
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
