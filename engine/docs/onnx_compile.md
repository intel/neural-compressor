# Compile an ONNX Model to Engine IR

The `Engine`  as the backend of `Neural_Compressor` support frozen static graph model from two deep learning framework (`TensorFlow` and `ONNX`) for now. The image below shows the workflow of how the `Engine` compile framework model to its own intermediate representation (IR). The `Loader` is used to load models from different deep learning framework. Then the `Extractors` would extract operations of the origin model and compose the engine graph. Next, the `Subgraph matcher`  implement pattern fusion for accelerating inference. In the end, the `Emitter` saves the final intermediate graph on the disk as the format of `.yaml` and `.bin` files.

![](imgs/compile_workflow.png)

Here is one example show that how to use `Engine` to compile `ONNX` model to `Engine` intermediate representations. In this example, we will compile  `distilbert_base`  model in `ONNX` framework on task `MRPC` to `Engine` IR.

## Prepare your environment


  ```shell
  # clone the neural_compressor repository
  git clone https://github.com/intel/neural-compressor.git
  cd <nc_folder>/examples/engine/nlp/mrpc/distilbert_base_uncased

  # use conda create new work environment
  conda create -n <your_env_name> python=3.7
  conda activate <your_env_name>

  # install necessary requirements
  pip install -r requirements.txt
  ```

## Prepare pretrained model

You can get the `distilbert_base` from [Hugging Face](https://huggingface.co/), and train it on `MRPC` task.


Train the `distilbert_base` and export the model

```shell
bash prepare_model.sh
```

Then you will get the `distilbert_base_uncased_mrpc.onnx` model in the folder.

>  **NOTE**: However, you can also choose not to train the model and just compile a `distilbert_base` model without task layers for quick use. But the engine ir can not be used for deployment or quantization later because it just output raw logits.

Here are the commands of get `distilbert_base` onnx model without `MRPC` task layer.

```shell
python -m transformers.onnx --model=distilbert-base-uncased distilbert-base-uncased/
```

Then you will get the `distilbert_base` model  `model.onnx`  without task layer in the `<./distilbert-base-uncased>` folder.

## Compile the distilbert_base model to Engine IR

```python
# import compile api form engine
from engine.compile import compile
# get the engine intermediate graph (if trained on MRPC task)
graph = compile("distilbert_base_uncased_mrpc.onnx")
# get the engine intermediate graph (if not trained on MRPC task)
graph = compile("distilbert-base-uncased/model.onnx")
# save the graph and get the final ir
# the yaml and bin file will stored in '<ir>' folder
graph.save()
# you can also set the ir output folder, like
graph.save('my_ir')
```
