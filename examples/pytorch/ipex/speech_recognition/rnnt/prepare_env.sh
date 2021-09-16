  #set -eo pipefail  
  set -x

  WORKDIR=`pwd`

  PATTERN='[-a-zA-Z0-9_]*='
  if [ $# -lt "0" ] ; then
      echo 'ERROR:'
      printf 'Please use following parameters:
      --code=<mlperf workload repo directory> 
      '
      exit 1
  fi

  for i in "$@"
  do
      case $i in
         --code=*)
              code=`echo $i | sed "s/${PATTERN}//"`;;
          *)
              echo "Parameter $i not recognized."; exit 1;;
      esac
  done

  if [ -d $code ];then
     REPODIR=$code
  fi 

  echo "Install dependencies" 
  pip install sklearn onnx tqdm lark-parser
  pip install -e git+https://github.com/mlperf/logging@0.7.0-rc2#egg=mlperf-logging
  conda install ninja pyyaml setuptools cmake cffi typing --yes
  conda install intel-openmp mkl mkl-include numpy --no-update-deps --yes
  conda install -c conda-forge gperftools --yes
  conda install jemalloc=5.0.1 --yes
  pip install opencv-python absl-py opencv-python-headless intel-openmp

  echo "Install libraries"
  mkdir $WORKDIR/local
  export install_dir=$WORKDIR/local
  cd $WORKDIR && mkdir third_party
  wget https://ftp.osuosl.org/pub/xiph/releases/flac/flac-1.3.2.tar.xz -O third_party/flac-1.3.2.tar.xz
  cd third_party && tar xf flac-1.3.2.tar.xz && cd flac-1.3.2
  ./configure --prefix=$install_dir && make && make install

  cd $WORKDIR
  wget https://sourceforge.net/projects/sox/files/sox/14.4.2/sox-14.4.2.tar.gz -O third_party/sox-14.4.2.tar.gz
  cd third_party && tar zxf sox-14.4.2.tar.gz && cd sox-14.4.2
  LDFLAGS="-L${install_dir}/lib" CFLAGS="-I${install_dir}/include" ./configure --prefix=$install_dir --with-flac && make &&    make install

  cd $WORKDIR
  wget http://www.mega-nerd.com/libsndfile/files/libsndfile-1.0.28.tar.gz -O third_party/libsndfile-1.0.28.tar.gz
  cd third_party && tar zxf libsndfile-1.0.28.tar.gz && cd libsndfile-1.0.28
  ./configure --prefix=$install_dir && make && make install

  echo "Install pytorch/ipex"
  export LD_LIBRARY_PATH=$WORKDIR/local/lib:$LD_LIBRARY_PATH

  bash prepare_loadgen.sh ${WORKDIR}

  echo "Install dependencies for pytorch_SUT.py"
  pip install toml unidecode inflect librosa

  set +x
