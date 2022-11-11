Distillation
============

1. [Introduction](#introduction)

    1.1. [Knowledge Distillation](#knowledge-distillation)

    1.2. [Intermediate Layer Knowledge Distillation](#intermediate-layer-knowledge-distillation)

2. [Distillation Support Matrix](#distillation-support-matrix)
3. [Get Started with Distillation API ](#get-started-with-distillation-api)
4. [Examples](#examples)

## Introduction

Distillation is one of popular approaches of network compression, which transfers knowledge from a large model to a smaller one without loss of validity. As smaller models are less expensive to evaluate, they can be deployed on less powerful hardware (such as a mobile device). Graph shown below is the workflow of the distillation, the teacher model will take the same input that feed into the student model to produce the output that contains knowledge of the teacher model to instruct the student model.
<br>

<img src="./imgs/Distillation_workflow.png" alt="Architecture" width=700 height=300>

Intel® Neural Compressor supports Knowledge Distillation and Intermediate Layer Knowledge Distillation algorithms.

### Knowledge Distillation
Knowledge distillation is proposed in [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531). It leverages the logits (the input of softmax in the classification tasks) of teacher and student model to minimize the the difference between their predicted class distributions, this can be done by minimizing the below loss function. 

$$L_{KD} = D(z_t, z_s)$$

Where $D$ is a distance measurement, e.g. Euclidean distance and Kullback–Leibler divergence, $z_t$ and $z_s$ are the logits of teacher and student model, or predicted distributions from softmax of the logits in case the distance is measured in terms of distribution.

### Intermediate Layer Knowledge Distillation

There are more information contained in the teacher model beside its logits, for example, the output features of the teacher model's intermediate layers often been used to guide the student model, as in [Patient Knowledge Distillation for BERT Model Compression](https://arxiv.org/pdf/1908.09355) and [MobileBERT: a Compact Task-Agnostic BERT for Resource-Limited Devices](https://arxiv.org/abs/2004.02984). The general loss function for this approach can be summarized as follow.

$$L_{KD} = \sum\limits_i D(T_t^{n_i}(F_t^{n_i}), T_s^{m_i}(F_s^{m_i}))$$

Where $D$ is a distance measurement as before, $F_t^{n_i}$ the output feature of the $n_i$'s layer of the teacher model, $F_s^{m_i}$ the output feature of the $m_i$'s layer of the student model. Since the dimensions of $F_t^{n_i}$ and $F_s^{m_i}$ are usually different, the transformations $T_t^{n_i}$ and $T_s^{m_i}$ are needed to match dimensions of the two features. Specifically, the transformation can take the forms like identity, linear transformation, 1X1 convolution etc.

## Distillation Support Matrix

|Distillation Algorithm                          |PyTorch   |TensorFlow |
|------------------------------------------------|:--------:|:---------:|
|Knowledge Distillation                          |&#10004;  |&#10004;   |
|Intermediate Layer Knowledge Distillation       |&#10004;  |Will be supported|

## Get Started with Distillation API 

Simplest launcher code if training behavior is defined in user-defined yaml.

```python
from neural_compressor.experimental import Distillation, common
distiller = Distillation('/path/to/user/yaml')
distiller.student_model = student_model
distiller.teacher_model = teacher_model
model = distiller.fit()
```
Distillation class also support DistillationConf class as it's argument.

```python
from neural_compressor.experimental import Distillation, common
from neural_compressor.conf.config import DistillationConf
conf = DistillationConf('/path/to/user/yaml')
distiller = Distillation(conf)
distiller.student_model = student_model
distiller.teacher_model = teacher_model
model = distiller.fit()
```

User can pass the customized training/evaluation functions to `Distillation` for flexible scenarios. In this case, distillation process can be done by pre-defined hooks in Neural Compressor. User needs to put those hooks inside the training function.

Neural Compressor defines several hooks for user pass

```
on_train_begin() : Hook executed before training begins
on_after_compute_loss(input, student_output, student_loss) : Hook executed after each batch inference of student model
on_epoch_end() : Hook executed at each epoch end
```

Following section shows how to use hooks in user pass-in training function which is part of example from BlendCNN distillation:

```python
def train_func(model):
    distiller.on_train_begin()
    for nepoch in range(epochs):
        model.train()
        cnt = 0
        loss_sum = 0.
        iter_bar = tqdm(train_dataloader, desc='Iter (loss=X.XXX)')
        for batch in iter_bar:
            teacher_logits, input_ids, segment_ids, input_mask, target = batch
            cnt += 1
            output = model(input_ids, segment_ids, input_mask)
            loss = criterion(output, target)
            loss = distiller.on_after_compute_loss(
                {'input_ids':input_ids, 'segment_ids':segment_ids, 'input_mask':input_mask},
                output,
                loss,
                teacher_logits)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if cnt >= iters:
                break
        print('Average Loss: {}'.format(loss_sum / cnt))
        distiller.on_epoch_end()
...
```

In this case, the launcher code is like the following:

```python
from neural_compressor.experimental import Distillation, common
from neural_compressor.experimental.common.criterion import PyTorchKnowledgeDistillationLoss
distiller = Distillation(args.config)
distiller.student_model = model
distiller.teacher_model = teacher
distiller.criterion = PyTorchKnowledgeDistillationLoss()
distiller.train_func = train_func
model = distiller.fit()
```

## Examples

[Distillation Examples](../examples/README.md#distillation)
<br>
[Distillation Examples Results](./validated_model_list.md#validated-knowledge-distillation-examples)
