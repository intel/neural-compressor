# distributed example on QAT

Use following command to enable distributed QAT demo

## Installation
```Shell
pip install -r requirements.txt
```

## Run
```
OMP_NUM_THREADS=24 horovodrun -np <num_of_processes> -H <hosts> python main_buildin.py -t -a resnet50 --pretrained --config ./conf_buildin.yaml
```
