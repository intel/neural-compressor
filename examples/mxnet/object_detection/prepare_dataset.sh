#!/bin/bash
# set -x

# DATA_DIR="/home/.mxnet/datasets"
DATA_NAME='voc'
coco_val_data_url="http://images.cocodataset.org/zips/val2017.zip"
coco_annotations_url="http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
voc_data_url="http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar"

help()
{
   cat <<- EOF

   Desc: Prepare dataset for MXNet Object Detection.

   -h --help              help info

   --dataset              set dataset category, voc or coco, default is voc.

   --data_path            directory of the download dataset, default is: /home/.mxnet/datasets/

EOF
   exit 0
}

function main {

  init_params "$@"

}

# init params
function init_params {

  for var in "$@"
  do
    case $var in
      --data_path=*)
          DATA_DIR=$(echo $var |cut -f2 -d=)
      ;;
      --dataset=*)
          DATA_NAME=$(echo $var |cut -f2 -d=)
          DATA_PATH=${DATA_DIR}/${DATA_NAME}
          if [ ${DATA_NAME} = "voc" ]; then
            download_voc_dataset
          fi
          if [ ${DATA_NAME} = "coco" ]; then
            download_coco_dataset
          fi
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
function download_coco_dataset {

  if [ ! -d ${DATA_PATH} ]; then
    mkdir -p ${DATA_PATH}
  fi

  cd ${DATA_PATH}
  if [ ! -f ${DATA_NAME} ]; then
    wget ${coco_val_data_url}
    wget ${coco_annotations_url}
    unzip "*.zip"
  else
    echo "Dataset ${DATA_NAME} is exist!"
  fi
  cd ../

}

# download_dataset
function download_voc_dataset {

  if [ ! -d ${DATA_PATH} ]; then
    mkdir -p ${DATA_PATH}
  fi

  cd ${DATA_PATH}
  if [ ! -f ${DATA_NAME} ]; then
    wget ${voc_data_url}
    tar -xvf *.tar
  else
    echo "Dataset ${DATA_NAME} is exist!"
  fi
  cd ../

}

main "$@"