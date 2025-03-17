Code Migration from Intel Neural Compressor 1.X to Intel Neural Compressor 2.X
===============

Intel Neural Compressor is a powerful open-source Python library that runs on Intel CPUs and GPUs, which delivers unified interfaces across multiple deep-learning frameworks for popular network compression technologies such as quantization, pruning, and knowledge distillation. We conduct several changes to our old APIs based on Intel Neural Compressor 1.X to make it more user-friendly and convenient for using. Just some simple steps could upgrade your code from Intel Neural Compressor 1.X to Intel Neural Compressor 2.X.

> It should be noted that we will stop to support Intel Neural Compressor 1.X in the future.

1. [Quantization](#model-quantization)
2. [Pruning](#pruning)
3. [Distillation](#distillation)
4. [Mix-Precision](#mix-precision)
5. [Orchestration](#orchestration)
6. [Benchmark](#benchmark)
7. [Examples](#examples)

## Model Quantization

Model Quantization is a very popular deep learning model optimization technique designed for improving the speed of model inference, which is a fundamental function in Intel Neural Compressor. There are two model quantization methods, Quantization Aware Training (QAT) and Post-training Quantization (PTQ). Our tool has provided comprehensive supports for these two kinds of model quantization methods in both Intel Neural Compressor 1.X and Intel Neural Compressor 2.X.

### Post-training Quantization

Post-training Quantization is the most easy way to quantize the model from FP32 into INT8 format offline. 

**Quantization with Intel Neural Compressor 1.X**

In Intel Neural Compressor 1.X, we resort to a `conf.yaml` to inject the config of the quantization settings.

```python
# main.py

# Basic settings of the model and the experimental settings for GLUE tasks.
model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
val_dataset = ...
val_dataloader = torch.utils.data.Dataloader(
    val_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.workers,
    ping_memory=True,
)


def eval_func(model): ...


# Quantization code
from neural_compressor.experimental import Quantization, common

calib_dataloader = eval_dataloader
quantizer = Quantization("conf.yaml")
quantizer.eval_func = eval_func
quantizer.calib_dataloader = calib_dataloader
quantizer.model = common.Model(model)
model = quantizer.fit()

from neural_compressor.utils.load_huggingface import save_for_huggingface_upstream

save_for_huggingface_upstream(model, tokenizer, output_dir)
```

We formulate the `conf.yaml` as in (https://github.com/intel/neural-compressor/blob/master/neural_compressor/template/ptq.yaml)

**Quantization with Intel Neural Compressor 2.X**

In Intel Neural Compressor 2.X, we integrate the `conf.yaml` into `main.py` to save the user's effort to write the `conf.yaml`, that most of config information could be set via the `PostTrainingQuantConfig`. The corresponding information should be written as follows (Note: the corresponding names of the parameters in 1.X yaml file are attached in the comment.),
```python
from neural_compressor.config import PostTrainingQuantConfig, TuningCriterion, AccuracyCriterion

PostTrainingQuantConfig(
    ## model: this parameter does not need to specially be defined;
    backend="default",  # framework: set as "default" when framework was tensorflow, pytorch, pytorch_fx, onnxrt_integer and onnxrt_qlinear. Set as "ipex" when framework was pytorch_ipex, mxnet is currently unsupported;
    inputs="image_tensor",  # input: same as in the conf.yaml;
    outputs="num_detections,detection_boxes,detection_scores,detection_classes",  # output: same as in the conf.yaml;
    device="cpu",  # device: same as in the conf.yaml;
    approach="static",  # approach: set as "static" when approach was "post_training_static_quant". Set as "dynamic" when approach was "post_training_dynamic_quant";
    ## recipes: this parameter does not need to specially be defined;
    calibration_sampling_size=[1000, 2000],  # sampling_size: same as in the conf.yaml;
    ## transform: this parameter does not need to specially be defined;
    ## model_wise: this parameter does not need to specially be defined;
    op_name_dict=op_dict,  # op_wise: same as in the conf.yaml;
    ## evaluation: these parameters do not need to specially be defined;
    strategy="basic",  # tuning.strategy.name: same as in the conf.yaml;
    ## tuning.strategy.sigopt_api_token, tuning.strategy.sigopt_project_id and tuning.strategy.sigopt_experiment_name do not need to specially defined;
    objective="performance",  # tuning.objective: same as in the conf.yaml;
    performance_only=False,  # tuning.performance_only: same as in the conf.yaml;
    tuning_criterion=tuning_criterion,
    accuracy_criterion=accuracy_criterion,
    ## tuning.random_seed and tuning.tensorboard: these parameters do not need to specially be defined;
)

accuracy_criterion = AccuracyCriterion(
    tolerable_loss=0.01,  # relative: same as in the conf.yaml;
)
tuning_criterion = TuningCriterion(
    timeout=0,  # timeout: same as in the conf.yaml;
    max_trials=100,  # max_trials: same as in the conf.yaml;
)
```

Following is a simple demo about how to quantize the model with PTQ in Intel Neural Compressor 2.X.

```python
# Basic settings of the model and the experimental settings for GLUE tasks.
model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
val_dataset = ...
val_dataloader = torch.utils.data.Dataloader(
    val_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.workers,
    ping_memory=True,
)


def eval_func(model): ...


# Quantization code
from neural_compressor.quantization import fit
from neural_compressor.config import PostTrainingQuantConfig, TuningCriterion

tuning_criterion = TuningCriterion(max_trials=600)
conf = PostTrainingQuantConfig(approach="static", tuning_criterion=tuning_criterion)
q_model = fit(model, conf=conf, calib_dataloader=eval_dataloader, eval_func=eval_func)
from neural_compressor.utils.load_huggingface import save_for_huggingface_upstream

save_for_huggingface_upstream(q_model, tokenizer, training_args.output_dir)
```

### Quantization Aware Training

Quantization aware training emulates inference-time quantization in the forward pass of the training process by inserting `fake quant` ops before those quantizable ops. With `quantization aware training`, all weights and activations are `fake quantized` during both the forward and backward passes of training. 

**Quantization with Intel Neural Compressor 1.X**

In Intel Neural Compressor 1.X, the difference between the QAT and PTQ is that we need to define the `train_func` in QAT to emulate the training process. The code is compiled as follows,

```python
# main.py

# Basic settings of the model and the experimental settings for GLUE tasks.
model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)


def eval_func(model): ...


def train_func(model): ...


trainer = Trainer(...)

# Quantization code
from neural_compressor.experimental import Quantization, common

quantizer = Quantization("conf.yaml")
quantizer.eval_func = eval_func
quantizer.q_func = train_func
quantizer.model = common.Model(model)
model = quantizer.fit()

from neural_compressor.utils.load_huggingface import save_for_huggingface_upstream

save_for_huggingface_upstream(model, tokenizer, output_dir)
```

Similar to PTQ, it requires a `conf.yaml` (https://github.com/intel/neural-compressor/blob/master/neural_compressor/template/qat.yaml) to define the quantization configuration in Intel Neural Compressor 1.X.

**Quantization with Intel Neural Compressor 2.X**

In Intel Neural Compressor 2.X, this `conf.yaml` is set via the `QuantizationAwareTrainingConfig`. The corresponding information should be written as follows (Note: the corresponding names of the parameters in 1.X yaml file are attached in the comment.)，

```python
from neural_compressor.config import QuantizationAwareTrainingConfig

QuantizationAwareTrainingConfig(
    ## model: this parameter does not need to specially be defined;
    backend="default",  # framework: set as "default" when framework was tensorflow, pytorch, pytorch_fx, onnxrt_integer and onnxrt_qlinear. Set as "ipex" when framework was pytorch_ipex, mxnet is currently unsupported;
    inputs="image_tensor",  # input: same as in the conf.yaml;
    outputs="num_detections,detection_boxes,detection_scores,detection_classes",  # output: same as in the conf.yaml;
    device="cpu",  # device: same as in the conf.yaml;
    ## approach: this parameter does not need to specially be defined;
    ## train: these parameters do not need to specially be defined;
    ## model_wise: this parameter does not need to specially be defined;
    op_name_dict=op_dict,  # op_wise: same as in the conf.yaml;
    ## evaluation: these parameters do not need to specially be defined;
    strategy="basic",  # tuning.strategy.name: same as in the conf.yaml;
    ## tuning.strategy.sigopt_api_token, tuning.strategy.sigopt_project_id and tuning.strategy.sigopt_experiment_name do not need to specially defined;
    relative=0.01,  # relative: same as in the conf.yaml;
    timeout=0,  # timeout: same as in the conf.yaml;
    max_trials=100,  # max_trials: same as in the conf.yaml;
    objective="performance",  # tuning.objective: same as in the conf.yaml;
    performance_only=False,  # tuning.performance_only: same as in the conf.yaml;
    ## tuning.random_seed and tuning.tensorboard: these parameters do not need to specially be defined;
)
```

In Intel Neural Compressor 2.X, we introduce a `compression manager`  to control the training process. It requires to insert a pair of hook `callbacks.on_train_begin` and `callbacks.on_train_end` at the begin of the training and the end of the training. Thus, the quantization code is updated as:

```python
# main.py

# Basic settings of the model and the experimental settings for GLUE tasks.
model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)


def eval_func(model): ...


def train_func(model): ...


trainer = Trainer(...)

# Quantization code
from neural_compressor.training import prepare_compression
from neural_compressor.config import QuantizationAwareTrainingConfig

conf = QuantizationAwareTrainingConfig()
compression_manager = prepare_compression(model, conf)
compression_manager.callbacks.on_train_begin()
trainer.train()
compression_manager.callbacks.on_train_end()

from neural_compressor.utils.load_huggingface import save_for_huggingface_upstream

save_for_huggingface_upstream(compression_manager.model, tokenizer, training_args.output_dir)
```

## Pruning

Neural network pruning (briefly known as pruning or sparsity) is one of the most promising model compression techniques. It removes the least important parameters in the network and achieves compact architectures with minimal accuracy drop and maximal inference acceleration.

**Pruning with Intel Neural Compressor 1.X**

In Intel Neural Compressor 1.X, the Pruning config is still defined by an extra `conf.yaml`. The pruning code should be written as:

```python
from neural_compressor.experimental import Pruning, common

prune = Pruning("conf.yaml")
prune.model = model
prune.train_func = pruning_func
model = prune.fit()
```

The `conf.yaml` is written as (https://github.com/intel/neural-compressor/blob/master/neural_compressor/template/pruning.yaml).

The pruning code requires the user to insert a series of pre-defined hooks in the training function to activate the pruning with Intel Neural Compressor. The pre-defined hooks are listed as follows,

```
on_epoch_begin(epoch) : Hook executed at each epoch beginning
on_step_begin(batch) : Hook executed at each batch beginning
on_step_end() : Hook executed at each batch end
on_epoch_end() : Hook executed at each epoch end
on_before_optimizer_step() : Hook executed after gradients calculated and before backward
```

Following is an example to show how we use the hooks in the training function,

```python
def pruning_func(model):
    for epoch in range(int(args.num_train_epochs)):
        model.train()
        prune.on_epoch_begin(epoch)
        for step, batch in enumerate(train_dataloader):
            prune.on_step_begin(step)
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
            # inputs['token_type_ids'] = batch[2]
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            if (step + 1) % args.gradient_accumulation_steps == 0:
                prune.on_before_optimizer_step()
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()

            prune.on_step_end()


...
```

**Pruning with Intel Neural Compressor 2.X**

In Intel Neural Compressor 2.X, the training process is activated by a `compression manager`. And the configuration information is included in the `WeightPruningConfig`. The corresponding config should be set via (Note: the corresponding names of the parameters in 1.X yaml file are attached in the comment.):
```python
from neural_compressor.config import WeightPruningConfig

WeightPruningConfig(
  ## model, device: these parameters do not need to specially be defined;
  start_step=0,             # start_epoch: same as in the conf.yaml;
  end_step=10,              # end_epoch: same as in the conf.yaml;
  pruning_frequency=2,      # frequency: same as in the conf.yaml;        
  ## dataloader, criterion, optimizer: these parameters do not need to specially be defined;
  target_sparsity=0.97,      # target_sparsity: same as in the conf.yaml;     
  pruning_configs: [pruning_config] # Pruner: same as in the conf.yaml;           
  ## evaluation and tuning: these parameters do not need to specially be defined;
)       
```

We also need to replace the hooks in the training code. The newly defined hooks are included in `compression manager` and listed as follows,

```python
on_train_begin() : Execute at the beginning of training phase.
on_epoch_begin(epoch) : Execute at the beginning of each epoch.
on_step_begin(batch) : Execute at the beginning of each batch.
on_step_end() : Execute at the end of each batch.
on_epoch_end() : Execute at the end of each epoch.
on_before_optimizer_step() : Execute before optimization step.
on_after_optimizer_step() : Execute after optimization step.
on_train_end() : Execute at the ending of training phase.
```

The final Pruning code is updated as follows,

```python
config = {  ## pruner
    "target_sparsity": 0.9,  # Target sparsity ratio of modules.
    "pruning_type": "snip_momentum",  # Default pruning type.
    "pattern": "4x1",  # Default pruning pattern.
    "op_names": ["layer1.*"],  # A list of modules that would be pruned.
    "excluded_op_names": ["layer3.*"],  # A list of modules that would not be pruned.
    "start_step": 0,  # Step at which to begin pruning.
    "end_step": 10,  # Step at which to end pruning.
    "pruning_scope": "global",  # Default pruning scope.
    "pruning_frequency": 1,  # Frequency of applying pruning.
    "min_sparsity_ratio_per_op": 0.0,  # Minimum sparsity ratio of each module.
    "max_sparsity_ratio_per_op": 0.98,  # Maximum sparsity ratio of each module.
    "sparsity_decay_type": "exp",  # Function applied to control pruning rate.
    "pruning_op_types": ["Conv", "Linear"],  # Types of op that would be pruned.
}

from neural_compressor.training import prepare_compression, WeightPruningConfig

##setting configs
pruning_configs = [
    {"op_names": ["layer1.*"], "pattern": "4x1"},
    {"op_names": ["layer2.*"], "pattern": "1x1", "target_sparsity": 0.5},
]
configs = WeightPruningConfig(
    pruning_configs=pruning_configs,
    target_sparsity=config.target_sparsity,
    pattern=config.pattern,
    pruning_frequency=config.pruning_frequency,
    start_step=config.start_step,
    end_step=config.end_step,
    pruning_type=config.pruning_type,
)
compression_manager = prepare_compression(model=model, confs=configs)
compression_manager.callbacks.on_train_begin()  ## insert hook
for epoch in range(num_train_epochs):
    model.train()
    for step, batch in enumerate(train_dataloader):
        compression_manager.callbacks.on_step_begin(step)  ## insert hook
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        compression_manager.callbacks.on_before_optimizer_step()  ## insert hook
        optimizer.step()
        compression_manager.callbacks.on_after_optimizer_step()  ## insert hook
        lr_scheduler.step()
        model.zero_grad()
...
compression_manager.callbacks.on_train_end()
```

## Distillation

Distillation is one of popular approaches of network compression, which transfers knowledge from a large model to a smaller one without loss of validity. As smaller models are less expensive to evaluate, they can be deployed on less powerful hardware (such as a mobile device).

**Distillation with Intel Neural Compressor 1.X**

Intel Neural Compressor distillation API is defined under `neural_compressor.experimental.Distillation`, which takes a user yaml file `conf.yaml` as input.

```yaml
    distillation:
    train:                    # optional. No need if user implements `train_func` and pass to `train_func` attribute of pruning instance.
        start_epoch: 0
        end_epoch: 10
        iteration: 100
        
        dataloader:
          batch_size: 256
        criterion:
          KnowledgeDistillationLoss:
              temperature: 1.0
              loss_types: ['CE', 'KL']
              loss_weights: [0.5, 0.5]
        optimizer:
          SGD:
              learning_rate: 0.1
              momentum: 0.9
              weight_decay: 0.0004
              nesterov: False
    evaluation:                              # optional. required if user doesn't provide eval_func in neural_compressor.Quantization.
    accuracy:                              # optional. required if user doesn't provide eval_func in neural_compressor.Quantization.
        metric:
        topk: 1                            # built-in metrics are topk, map, f1, allow user to register new metric.
        dataloader:
        batch_size: 256
        dataset:
            ImageFolder:
            root: /path/to/imagenet/val
        transform:
            RandomResizedCrop:
            size: 224
            RandomHorizontalFlip:
            ToTensor:
            Normalize:
            mean: [0.485, 0.456, 0.406]
            std: [0.229, 0.224, 0.225] 
```

We insert a series of pre-defined hooks to activate the training process of distillation,

```python
def train_func(model):
    distiller.on_train_begin()
    for nepoch in range(epochs):
        model.train()
        cnt = 0
        loss_sum = 0.0
        iter_bar = tqdm(train_dataloader, desc="Iter (loss=X.XXX)")
        for batch in iter_bar:
            teacher_logits, input_ids, segment_ids, input_mask, target = batch
            cnt += 1
            output = model(input_ids, segment_ids, input_mask)
            loss = criterion(output, target)
            loss = distiller.on_after_compute_loss(
                {"input_ids": input_ids, "segment_ids": segment_ids, "input_mask": input_mask},
                output,
                loss,
                teacher_logits,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if cnt >= iters:
                break
        print("Average Loss: {}".format(loss_sum / cnt))
        distiller.on_epoch_end()


from neural_compressor.experimental import Distillation, common
from neural_compressor.experimental.common.criterion import PyTorchKnowledgeDistillationLoss

distiller = Distillation(conf.yaml)
distiller.student_model = model
distiller.teacher_model = teacher
distiller.criterion = PyTorchKnowledgeDistillationLoss()
distiller.train_func = train_func
model = distiller.fit()
```

**Distillation with Intel Neural Compressor 2.X**

The new distillation API also introduce `compression manager` to conduct the training process. We continuously use the hooks in `compression manager` to activate the distillation process. We replace the `conf.yaml` with `DistillationConfig` API and clean the unnecessary parameters (Note: the corresponding names of the parameters in 1.X yaml file are attached in the comment.). The dataloader is directly inputted via `train_fun`. The updated code is shown as follows,

```python
from neural_compressor.config import DistillationConfig

DistillationConfig(
    criterion=KnowledgeDistillationLoss,  # criterion: same as in the conf.yaml;
    optimizer=SGD,  # optimizer: same as in the conf.yaml;
)
```

The newly updated distillation code is shown as follows,
```python
def training_func(model, dataloader):
    compression_manager.on_train_begin()
    for epoch in range(epochs):
        compression_manager.on_epoch_begin(epoch)
        for i, batch in enumerate(dataloader):
            compression_manager.on_step_begin(i)
            ......
            output = model(batch)
            loss = ......
            loss = compression_manager.on_after_compute_loss(batch, output, loss)
            loss.backward()
            compression_manager.on_before_optimizer_step()
            optimizer.step()
            compression_manager.on_step_end()
        compression_manager.on_epoch_end()
    compression_manager.on_train_end()

from neural_compressor.training import prepare_compression
from neural_compressor.config import DistillationConfig, SelfKnowledgeDistillationLossConfig

distil_loss = SelfKnowledgeDistillationLossConfig()
conf = DistillationConfig(teacher_model=model, criterion=distil_loss)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
compression_manager = prepare_compression(model, conf)
model = compression_manager.model

model = training_func(model, dataloader)
eval_func(model)
```

## Mix Precision

The recent growth of Deep Learning has driven the development of more complex models that require significantly more compute and memory capabilities. Mixed precision training and inference using low precision formats have been developed to reduce compute and bandwidth requirements. Intel Neural Compressor supports BF16 + FP32 mixed precision conversion by MixedPrecision API

**Mix Precision with Intel Neural Compressor 1.X**

The user can add dataloader and metric in `conf.yaml` to execute evaluation.
```python
from neural_compressor.experimental import MixedPrecision, common

dataset = Dataset()
converter = MixedPrecision("conf.yaml")
converter.metric = Metric()
converter.precisions = "bf16"
converter.eval_dataloader = common.DataLoader(dataset)
converter.model = "./model.pb"
output_model = converter()
```

The configuration `conf.yaml` should be,
```yaml
model:
  name: resnet50_v1
  framework: tensorflow
  inputs: image_tensor
  outputs: num_detections,detection_boxes,detection_scores,detection_classes

device: cpu

mixed_precision:
  precisions: 'bf16'  

evaluation:
  ...

tuning:
  accuracy_criterion:
    relative:  0.01                                  # optional. default value is relative, other value is absolute. this example allows relative accuracy loss: 1%.
  objective: performance                             # optional. objective with accuracy constraint guaranteed. default value is performance. other values are modelsize and footprint.

  exit_policy:
    timeout: 0                                       # optional. tuning timeout (seconds). default value is 0 which means early stop. combine with max_trials field to decide when to exit.
    max_trials: 100                                  # optional. max tune times. default value is 100. combine with timeout field to decide when to exit.
    performance_only: False                          # optional. max tune times. default value is False which means only generate fully quantized model.
  random_seed: 9527                                  # optional. random seed for deterministic tuning.

```

**Mix Precision with Intel Neural Compressor 2.X**

In 2.X version, we integrate the config information in `MixedPrecisionConfig`, leading to the updates in the code as follows (Note: the corresponding names of the parameters in 1.X yaml file are attached in the comment.),

```python
from neural_compressor.config import MixedPrecisionConfig, TuningCriterion, AccuracyCriterion

MixedPrecisionConfig(
    ## model: this parameter does not need to specially be defined;
    backend="default",  # framework: set as "default" when framework was tensorflow, pytorch, pytorch_fx, onnxrt_integer and onnxrt_qlinear. Set as "ipex" when framework was pytorch_ipex, mxnet is currently unsupported;
    inputs="image_tensor",  # input: same as in the conf.yaml;
    outputs="num_detections,detection_boxes,detection_scores,detection_classes",  # output: same as in the conf.yaml;
    device="cpu",  # device: same as in the conf.yaml;
    tuning_criterion=tuning_criterion,
    accuracy_criterion=accuracy_criterion,
    ## tuning.random_seed and tuning.tensorboard: these parameters do not need to specially be defined;
)

accuracy_criterion = AccuracyCriterion(
    tolerable_loss=0.01,  # relative: same as in the conf.yaml;
)
tuning_criterion = TuningCriterion(
    timeout=0,  # timeout: same as in the conf.yaml;
    max_trials=100,  # max_trials: same as in the conf.yaml;
)
```

The update demo is shown as follows,

```python
from neural_compressor import mix_precision
from neural_compressor.config import MixedPrecisionConfig

conf = MixedPrecisionConfig()

converted_model = mix_precision.fit(model, config=conf)
converted_model.save("./path/to/save/")
```

## Orchestration

Intel Neural Compressor supports arbitrary meaningful combinations of supported optimization methods under one-shot or multi-shot, such as pruning during quantization-aware training, or pruning and then post-training quantization, pruning and then distillation and then quantization.

**Orchestration with Intel Neural Compressor 1.X**

Intel Neural Compressor 1.X mainly relies on a `Scheduler` class to automatically pipeline execute model optimization with one shot or multiple shots way.

Following is an example how to set the `Scheduler` for Orchestration process. If the user wants to execute the pruning and quantization-aware training with one-shot way,
```python
from neural_compressor.experimental import Quantization, Pruning, Scheduler

prune = Pruning(prune_conf.yaml)
quantizer = Quantization(quantization_aware_training_conf.yaml)
scheduler = Scheduler()
scheduler.model = model
combination = scheduler.combine(prune, quantizer)
scheduler.append(combination)
opt_model = scheduler.fit()
```
The user needs to write the `prune_conf.yaml` and the `quantization_aware_training_conf.yaml` as aforementioned to configure the experimental settings.

**Orchestration with Intel Neural Compressor 2.X**

Intel Neural Compressor 2.X introduces `compression manager` to schedule the training process. The `conf.yaml` should follows in the previous sessions to set the configuration information. It also requires to inset the hooks from `compression manager` to control the training flow. Therefore, the code should be updated as,

```python
from neural_compressor.training import prepare_compression
from neural_compressor.config import DistillationConfig, KnowledgeDistillationLossConfig, WeightPruningConfig
distillation_criterion = KnowledgeDistillationLossConfig()
d_conf = DistillationConfig(model, distillation_criterion)
p_conf = WeightPruningConfig()
compression_manager = prepare_compression(model=model, confs=[d_conf, p_conf])

compression_manager.callbacks.on_train_begin()
train_loop:
    compression_manager.on_train_begin()
    for epoch in range(epochs):
        compression_manager.on_epoch_begin(epoch)
        for i, batch in enumerate(dataloader):
            compression_manager.on_step_begin(i)
            ......
            output = model(batch)
            loss = ......
            loss = compression_manager.on_after_compute_loss(batch, output, loss)
            loss.backward()
            compression_manager.on_before_optimizer_step()
            optimizer.step()
            compression_manager.on_step_end()
        compression_manager.on_epoch_end()
    compression_manager.on_train_end()

model.save('./path/to/save')
```

## Benchmark

The benchmarking feature of Neural Compressor is used to measure the model performance with the objective settings. The user can get the performance of the models between the float32 model and the quantized low precision model in a same scenario.

**Benchmark with Intel Neural Compressor 1.X**

In Intel Neural Compressor 1.X requires the user to define the experimental settings for low precision model and FP32 model with a `conf.yaml`.
```yaml
version: 1.0

model:                                               # mandatory. used to specify model specific information.
  name: ssd_mobilenet_v1                             # mandatory. the model name.
  framework: tensorflow                              # mandatory. supported values are tensorflow, pytorch, pytorch_fx, pytorch_ipex, onnxrt_integer, onnxrt_qlinear or mxnet; allow new framework backend extension.
  inputs: image_tensor                               # optional. inputs and outputs fields are only required in tensorflow.
  outputs: num_detections,detection_boxes,detection_scores,detection_classes

device: cpu                                          # optional. default value is cpu. other value is gpu.

evaluation:                                          # optional. used to config evaluation process.
  performance:                                       # optional. used to benchmark performance of passing model.
    warmup: 10
    iteration: 100
    configs:
      cores_per_instance: 4
      num_of_instance: 7
      inter_num_of_threads: 1
      intra_num_of_threads: 4
    dataloader:
      dataset:
        dummy:
          shape: [[128, 3, 224, 224], [128, 1, 1, 1]]
```

And then, the user can get the accuracy with,
```python
dataset = Dataset()  #  dataset class that implement __getitem__ method or __iter__ method
from neural_compressor.experimental import Benchmark, common
from neural_compressor.conf.config import BenchmarkConf

conf = BenchmarkConf(config.yaml)
evaluator = Benchmark(conf)
evaluator.dataloader = common.DataLoader(dataset, batch_size=batch_size)
# user can also register postprocess and metric, this is optional
evaluator.postprocess = common.Postprocess(postprocess_cls)
evaluator.metric = common.Metric(metric_cls)
results = evaluator()
```

**Benchmark with Intel Neural Compressor 2.X**

In Intel Neural Compressor 2.X, we optimize the code to make it simple and clear for the user. We replace `conf.yaml` with `BenchmarkConfig`. The corresponding information should be defined as (Note: the corresponding names of the parameters in 1.X yaml file are attached in the comment.):
```python
from neural_compressor.config import BenchmarkConfig

BenchmarkConfig(
    ## model: this parameter does not need to specially be defined;
    backend="default",  # framework: set as "default" when framework was tensorflow, pytorch, pytorch_fx, onnxrt_integer and onnxrt_qlinear. Set as "ipex" when framework was pytorch_ipex, mxnet is currently unsupported;
    inputs="image_tensor",  # input: same as in the conf.yaml;
    outputs="num_detections,detection_boxes,detection_scores,detection_classes",  # output: same as in the conf.yaml;
    device="cpu",  # device: same as in the conf.yaml;
    warmup=10,  # warmup: same as in the conf.yaml;
    iteration=100,  # iteration: same as in the conf.yaml;
    cores_per_instance=4,  # cores_per_instance: same as in the conf.yaml;
    num_of_instance=7,  # num_of_instance: same as in the conf.yaml;
    inter_num_of_threads=1,  # inter_num_of_threads: same as in the conf.yaml;
    intra_num_of_threads=4,  # intra_num_of_threads: same as in the conf.yaml;
    ## dataloader: this parameter does not need to specially be defined;
)
```

The new example in Intel Neural Compressor 2.X should be updated as,

```python
from neural_compressor.config import BenchmarkConfig
from neural_compressor.benchmark import fit

conf = BenchmarkConfig(warmup=10, iteration=100, cores_per_instance=4, num_of_instance=7)
fit(model="./int8.pb", config=conf, b_dataloader=eval_dataloader)
```


## Examples

User could refer to [examples](https://github.com/intel/neural-compressor/blob/master/examples/README.md) for more details about the migration from Intel Neural Compressor 1.X to Intel Neural Compressor 2.X.
