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
    if [ "${topology}" = "efficientnet_b0" ];then
        conf_yaml=conf_efficientnet_b0.yaml
    elif [ "${topology}" = "mobilenetv3_rw" ]; then
        conf_yaml=conf_mobilenetv3_rw.yaml
        sed -i "/relative:/s|relative:.*|relative: 0.02|g" $conf_yaml
    fi
    sed -i "/\/path\/to\/calibration\/dataset/s|root:.*|root: $dataset_location/train|g" $conf_yaml
    sed -i "/\/path\/to\/evaluation\/dataset/s|root:.*|root: $dataset_location/val|g" $conf_yaml
    extra_cmd=""
    if [ -n "$output_model" ];then
        extra_cmd = "--tuned_checkpoint ${output_model}"
    fi
    extra_cmd=$extra_cmd" ${dataset_location}"

    python validate.py \
            --pretrained \
            --model $topology \
            -b 30 \
            --no-cuda \
            --tune \
            ${extra_cmd}

}

main "$@"
