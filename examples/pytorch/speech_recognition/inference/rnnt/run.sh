#/bin/bash

set -euo pipefail

work_dir=/export/b07/ws15dgalvez/mlperf-rnnt-librispeech
local_data_dir=$work_dir/local_data
librispeech_download_dir=$local_data_dir/LibriSpeech
stage=3

mkdir -p $work_dir $local_data_dir $librispeech_download_dir

install_dir=third_party/install
mkdir -p $install_dir
install_dir=$(readlink -f $install_dir)

set +u
source "$($CONDA_EXE info --base)/etc/profile.d/conda.sh"
set -u

# stage -1: install dependencies
if [[ $stage -le -1 ]]; then
    conda env create --force -v --file environment.yml

    set +u
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate mlperf-rnnt
    set -u

    # We need to convert .flac files to .wav files via sox. Not all sox installs have flac support, so we install from source.
    wget https://ftp.osuosl.org/pub/xiph/releases/flac/flac-1.3.2.tar.xz -O third_party/flac-1.3.2.tar.xz
    (cd third_party; tar xf flac-1.3.2.tar.xz; cd flac-1.3.2; ./configure --prefix=$install_dir && make && make install)

    wget https://sourceforge.net/projects/sox/files/sox/14.4.2/sox-14.4.2.tar.gz -O third_party/sox-14.4.2.tar.gz
    (cd third_party; tar zxf sox-14.4.2.tar.gz; cd sox-14.4.2; LDFLAGS="-L${install_dir}/lib" CFLAGS="-I${install_dir}/include" ./configure --prefix=$install_dir --with-flac && make && make install)

    (cd $(git rev-parse --show-toplevel)/loadgen; python setup.py install)
fi

export PATH="$install_dir/bin/:$PATH"

set +u
conda activate mlperf-rnnt
set -u

# stage 0: download model. Check checksum to skip?
if [[ $stage -le 0 ]]; then
  wget https://zenodo.org/record/3662521/files/DistributedDataParallel_1576581068.9962234-epoch-100.pt?download=1 -O $work_dir/rnnt.pt
fi

# stage 1: download data. This will hae a non-zero exit code if the
# checksum is incorrect.
if [[ $stage -le 1 ]]; then
  python pytorch/utils/download_librispeech.py \
         pytorch/utils/librispeech-inference.csv \
         $librispeech_download_dir \
         -e $local_data_dir
fi

if [[ $stage -le 2 ]]; then
  python pytorch/utils/convert_librispeech.py \
      --input_dir $librispeech_download_dir/dev-clean \
      --dest_dir $local_data_dir/dev-clean-wav \
      --output_json $local_data_dir/dev-clean-wav.json
fi

if [[ $stage -le 3 ]]; then
  for backend in pytorch; do
    for accuracy in "--accuracy" ""; do
      for scenario in SingleStream Offline Server; do
        log_dir=${work_dir}/${scenario}_${backend}
        if [ ! -z ${accuracy} ]; then
          log_dir+=_accuracy
        fi
        log_dir+=rerun

        python run.py --backend pytorch \
               --dataset_dir $local_data_dir \
               --manifest $local_data_dir/dev-clean-wav.json \
               --pytorch_config_toml pytorch/configs/rnnt.toml \
               --pytorch_checkpoint $work_dir/rnnt.pt \
               --scenario ${scenario} \
               --backend ${backend} \
               --log_dir ${log_dir} \
               ${accuracy} &

      done
    done
  done
  wait
fi
