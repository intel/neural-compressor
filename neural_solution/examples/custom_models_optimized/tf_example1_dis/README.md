# Distributed version of the tf_example1

## Requirements
1. dataset/
2. model/
3. run.py

## Reference
(https://github.com/intel/neural-compressor/tree/master/examples/helloworld/tf_example1)
## Run
### Run Command

```shell
mpirun -np 3 -host localhost,localhost,localhost -map-by socket:pe=5 -mca btl_tcp_if_include 192.168.20.0/24 \
-x OMP_NUM_THREADS=5 --report-bindings python test.py --dataset_location=dataset --model_path=model
