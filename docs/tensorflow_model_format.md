Tensorflow Model Format Supporting in Intel® Low Precision Optimization Tool
===========================================

The Tensorflow has several ways to save the model for inference. Currently, Intel® Low Precision Optimization Tool supports the frozen pb(.pb), ckpt format and keras model saved via keras API.This doc will focus on the introduction for the ckpt and keras model usage on this tool.

# Ckpt Format

The checkpoint format usually contains of model description information captured from training phase.Typically, there're three kinds of files because it stores the graph structure separately from the variable values.
* **meta file** describes the saved graph structure
* **index file** it's a string-string immutable table to describe the mapping between tensor name and variables.
* **data file** save the values of all variables.

Currently, the tool requires the ckpt format which converted by [Intel Tensorflow 1.15](https://pypi.org/project/intel-tensorflow/) must contain graph structure description file and variables. It means the tool doesn't support ckpt format consists of variables only.


# Keras Format
The tool also supports the model saved as [Keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) format.

# How to use it.
We support the end user passing down the Ckpt or Keras model folder path into ilit tune() func as parameter besides frozen pb.
Below code snippet is the typical way to use with ilit. We only need to specify the model parameter equal to the model path.
```python
import ilit

at = ilit.Tuner(args.config)
q_model = at.tune(model, #the path to ckpt or keras model folder
                q_dataloader=infer,
                eval_func=infer.accuracy_check)
```

Once the tuning is over, the tool provides the interface *convert_pb_to_savedmodel* which located on ilit.adaptor.tf_utils.util module to convert the pb to keras model format for further deployment.