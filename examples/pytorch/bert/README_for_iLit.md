## Dependency
pytorch: commit: 24aac321718d58791c4e6b7cfa50788a124dae23
### note
the latest version of pytorch enabled INT8 layer_norm op, but the accuracy was regression. So you should tune BERT model on commit 24aac321718d58791c4e6b7cfa50788a124dae23.
## Installation
```
cd examples/pytorch/bert
python setup.py install
```

## Run the examples

```
./run_all.sh
```
## task
There are two dataset: glue and SQuAD.
10 task:"MRPC" "CoLA" "STS-B" "SST-2" "RTE" "SQuAD" "MRPC" "QNLI" "RTE" "CoLA"


