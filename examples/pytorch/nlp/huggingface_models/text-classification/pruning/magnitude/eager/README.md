### Prepare dataset
The [General Language Understanding Evaluation (GLUE) benchmark](https://gluebenchmark.com/) is a collection of nine sentence- or sentence-pair language understanding tasks for evaluating and analyzing natural language understanding systems.

Before running anyone of these GLUE tasks you should download the [GLUE data](https://gluebenchmark.com/tasks) by running [this script](https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e) and unpack it to some directory data_dir.
### Prepare environment
```shell
cd examples/pytorch/nlp/huggingface_models/pruning/magnitude/eager 
pip install -r requirements.txt
```

### Pruning
Pruning now support basic magnitude for distilbert and gradient sensitivity for bert-base:

- Enable magnitude pruning example:

```shell
bash run_pruning.sh --topology=distilbert_SST-2 --data_dir=path/to/dataset --output_model=path/to/output_model --config=path/to/conf.yaml
```
