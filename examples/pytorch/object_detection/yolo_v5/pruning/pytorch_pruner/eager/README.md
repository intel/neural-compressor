# PyTorch Pruner
## Intro
[**PyTorch Pruner**](https://github.com/intel/neural-compressor/tree/master/neural_compressor/experimental/pytorch_pruner) is an Intel® Neural Compressor build-in API which supports a wide range of pruning algorithms, Criteria, patterns as well as pruning schedulers. Features below are currently supported:
> algorithms: magnitude, snip, snip-momentum\
> patterns: NxM, N:M\
> pruning schedulers: iterative pruning scheduler, one-shot pruning scheduler.

## Usage
### Write a config yaml file
PyTorch pruner is developed based on [pruning](https://github.com/intel/neural-compressor/blob/master/neural_compressor/experimental/pruning.py), therefore most usages are identical. Our API reads in a yaml configuration file to define a Pruning object. Here is an yolov5s example of it:
```yaml
version: 1.0

model:
  name: "yolov5s6"
  framework: "pytorch"

pruning:
  approach:
    weight_compression_pytorch:
      # Global settings
      # if start step equals to end step, one-shot pruning scheduler is enabled. Otherwise the API automatically implements iterative pruning scheduler.
      start_step: 0 # step which pruning process begins
      end_step: 0 # step which pruning process ends
      not_to_prune_names: ["model.33.m.*"] # a global announcement of layers which you do not wish to prune. 
      prune_layer_type: ["Conv2d"] # the module type which you want to prune (Linear, Conv2d, etc.)
      target_sparsity: 0.8 # the sparsity you want the model to be pruned.
      max_sparsity_ratio_per_layer: 0.98 # the highest sparsity a layer can reach.

      pruners: # below each "Pruner" defines a pruning process for a group of layers. This enables us to apply different pruning methods for different layers in one model.
        # Local settings
        - !Pruner
            exclude_names: ["model.*.cv1", "model.*.cv2", "model.*.cv3"] # list of regular expressions, containing the layer names you wish not to be included in this pruner
            pattern: "oc_pattern_1x1" # pattern type, we support "NxM" and "N:M"
            update_frequency_on_step: 1000 # if use iterative pruning scheduler, this define the pruning frequency.
            prune_domain: "global" # one in ["global", "local"], refers to the score map is computed out of entire parameters or its corresponding layer's weight.
            prune_type: "snip_momentum" # pruning algorithms, refer to pytorch_pruner/pruner.py
            sparsity_decay_type: "exp" # ["linear", "cos", "exp", "cube"] ways to determine the target sparsity during iterative pruning.
        - !Pruner
            exclude_names: []
            pattern: "oc_pattern_4x1"
            update_frequency_on_step: 1000
            prune_domain: "global"
            prune_type: "snip_momentum"
            sparsity_decay_type: "exp"
```
Please be aware that when the keywords appear in both global and local settings, we select the **local** settings as priority.
### Coding template:
With a settled config file, we provide a template for implementing pytorch_pruner API:
```python
model = Model()
criterion = Criterion()
optimizer = Optimizer()
args = Args()

from neural_compressor.experimental.pytorch_pruner.pruning import Pruning

pruner = Pruning("path/to/your/config.yaml")
if opt.do_prune:
    start, end = nw, int(total_iterations)
else:
    start = nb * epochs + 1
    end = start
pruner.update_items_for_all_pruners(start_step=start, end_step=end)
pruner.model = model
pruner.on_train_begin()
for epoch in range(start_epoch, epochs):
    callbacks.run('on_train_epoch_start')
    model.train()
    for i, (imgs, targets, paths, _) in pbar:
        ni = i + nb * epoch  # number integrated batches (since train start)
        imgs = imgs.to(device, non_blocking=True).float() / 255
        # Forward
        with torch.cuda.amp.autocast(amp):
            pruner.on_step_begin(local_step=ni)
            pred = model(imgs)  # forward
            loss, loss_items = compute_loss(pred, targets.to(device))  # loss scaled by batch_size
        # Backward
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)  # unscale gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)  # clip gradients
        pruner.on_before_optimizer_step()
        scaler.step(optimizer)  # optimizer.step
        pruner.on_after_optimizer_step()
        scaler.update()
        optimizer.zero_grad()

    ...
```
For more usage, please refer to our example codes below.

## Examples
we have provided several pruning examples, which are trained on different datasets/tasks, use different sparsity patterns, etc. We are working on sharing our sparse models on HuggingFace.
### [yolo-coco](https://github.com/intel/neural-compressor/tree/master/examples/pytorch/object_detection/yolo_v5/pruning/pytorch_pruner/eager)
We can train a sparse model with NxM (2:4) pattern:
```python
python3 -m torch.distributed.run --nproc_per_node 2 --master_port='29500' \
        examples/pruning/yolov5/train_prune.py \
        --data examples/pruning/yolov5/data/coco.yaml \
        --hyp examples/pruning/yolov5/data/hyp.scratch-low.yaml \
        --pruning_config "./examples/pruning/yolov5/data/yolov5s6_prune.yaml" \
        --weights /path/to/dense_finetuned_model/ \
        --device 0,1 \
        --img 640 \
        --do_prune \
        --epochs 250 \
        --cooldown_epochs 150 \
        --batch-size 64 \
        --patience 0
```
We can also choose pruning with distillation(l2/kl):
```python
python3 -m torch.distributed.run --nproc_per_node 2 --master_port='29500' \
        ./examples/pruning/yolov5/train_prune.py \
        --data examples/pruning/yolov5/data/coco.yaml \
        --hyp examples/pruning/yolov5/data/hyp.scratch-low.yaml \
        --pruning_config "./examples/pruning/yolov5/data/yolov5s6_prune.yaml" \
        --device 0,1 \
        --img 640 \
        --do_prune \
        --dist_loss l2 \
        --temperature 10 \
        --epochs 250 \
        --cooldown_epochs 150 \
        --batch-size 64 \
        --patience 0
```
Dense model training is also supported as following (by setting --do_prune to False):
```python
python3 -m torch.distributed.run --nproc_per_node 2 --master_port='29500' \
        examples/pruning/yolov5/train_prune.py \
        --data examples/pruning/yolov5/data/coco.yaml \
        --hyp examples/pruning/yolov5/data/hyp.scratch-low.yaml \
        --pruning_config "./examples/pruning/yolov5/data/yolov5s6_prune.yaml" \
        --weights /path/to/dense_pretrain_model/ \
        --device 0,1 \
        --img 640 \
        --epochs 300 \
        --batch-size 64
```

## References
* [SNIP: Single-shot Network Pruning based on Connection Sensitivity](https://arxiv.org/abs/1810.02340)
* [Object detection at 200 Frames Per Second](https://arxiv.org/pdf/1805.06361.pdf)
