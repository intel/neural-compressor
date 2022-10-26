#!/bin/bash
set -x

function main {

  init_params "$@"
  run_tuning

}

# init params
function init_params {
  output_model=saved_results
  for var in "$@"
  do
    case $var in
      --topology=*)
          topology=$(echo $var |cut -f2 -d=)
      ;;
      --dataset_location=*)
          dataset_location=$(echo $var |cut -f2 -d=)
      ;;
      --input_model=*)
          input_model=$(echo $var |cut -f2 -d=)
      ;;
      --output_model=*)
          output_model=$(echo $var |cut -f2 -d=)
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
    if [ "mobilenet_v2" = "$topology" ];then
        sed -i "/relative:/s|relative:.*|relative: 0.02|g" conf.yaml
    fi
    if [ "resnet18_fx" = "$topology" ];then
        sed -i "/relative:/s|relative:.*|relative: 0.001|g" conf.yaml
    fi
    extra_cmd=""
    if [ -n "$output_model" ];then
        extra_cmd = $extra_cmd"--tuned_checkpoint ${output_model}"
    fi
    result=$(echo $topology | grep "ipex")
    if [[ "$result" != "" ]];then
        sed -i "/\/path\/to\/calibration\/dataset/s|root:.*|root: $dataset_location/train|g" conf_ipex.yaml
        sed -i "/\/path\/to\/evaluation\/dataset/s|root:.*|root: $dataset_location/val|g" conf_ipex.yaml
        extra_cmd=$extra_cmd" --ipex"
        topology=${topology%*${topology:(-5)}}
    else
        sed -i "/\/path\/to\/calibration\/dataset/s|root:.*|root: $dataset_location/train|g" conf.yaml
        sed -i "/\/path\/to\/evaluation\/dataset/s|root:.*|root: $dataset_location/val|g" conf.yaml
    fi
    extra_cmd=$extra_cmd" ${dataset_location}"

    python main.py \
            --pretrained \
            -t \
            -a $topology \
            -b 30 \
            ${extra_cmd}

}

main "$@"
