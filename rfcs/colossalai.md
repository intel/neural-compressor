## **Summary**

This RFC is used to unify the access interface to different devices (cpu and gpu) in the components of ColossalAI. This propose will also benefit on adding Intel XPU support to ColossalAI in the future.

## **Motivation**

As we know, the major features of ColossalAI are built on Nvidia GPU and Cuda package. This limits the scope of leveraging different device types to enable LLM by ColossalAI. 

for example, in `utils/cuda.py`, `context/parallel_context.py` and something else, there have had some seperate interfaces for other componenets to access `cpu` or `gpu` device. Besides that, there are also many internal components invoking `torch.cuda` explicitly.

We would like to propose a unified device access interface to provide not only Nvidia GPU support but also other device type support, like Intel X86 CPU and XPU.

## **Proposals**

*NOTE 1: Currently the proposal mainly focues on ColossalAI training part. The ColossalAI inference support is out of scope here.*

*NOTE 2: The RFC focus on pythonic API level only, the replacement of cuda kernels on cpu part and the corresponding upper layer features, like nn.optimizer, nn.layer, gemini and so on, are out of scope here. we plan to provide support in the other RFC/PRs.*

As ColossalAI training is designed to boost up NLP & LLM training speed by data and model parallel, it has had a central place to store the `execution context` in `core.global_context`. The first proposal of unifying the device access is to extend this `core.global_context` structure to get and set device related informations.

The `engine` and `trainer` user facing API will rely on it to copy tensor to the corresponding device the application is runing on. 

The details are something like below:

```
class ParallelContext(metaclass=SingletonMeta):
    ### existing methods
    ...

    ### new methods
    def set_device(self, device_ordinal=None):
    # set the current device
    ...

    def get_device_name(self):
    # get the name of device
    ...

    def to(self):
    # move or cast the parameter to specified device
    ...

    def device_count(self):
    # get the number of devices
    ...

    def synchronize(self, device_index=None):
    # communicate with specified device
    ...
    
    def random(self):
    # random number generator

    def stream(self):
    # get device stream

    # other device related methods, details will be included in PRs.
    ...
```

From user view, the training code is simpler than before as user doesn't need to care which device should be explicitly specified. The new logic will automatically move/cast tensors to the device user is using. For example, if the underneath hardware is Nvidia GPU, the tensor will be automatically go to CUDA device. If the underneath hardware is Intel X86 CPU, the tensor will automatically keep in CPU side. 

Blow is some sample codes to demonstrate this idea.

```
# The lines commented off are not needed as engine and trainer will automatically copy tensors from host to device.

for epoch in range(gpc.config.NUM_EPOCHS):
    # execute a training iteration
    engine.train()
    for img, label in train_dataloader:
        #img = img.cuda()
        #label = label.cuda()

        # set gradients to zero
        engine.zero_grad()

        # run forward pass
        output = engine(img)

        # compute loss value and run backward pass
        train_loss = engine.criterion(output, label)
        engine.backward(train_loss)

        # update parameters
        engine.step()

    # update learning rate
    lr_scheduler.step()

    # execute a testing iteration
    engine.eval()
    correct = 0
    total = 0
    for img, label in test_dataloader:
        #img = img.cuda()
        #label = label.cuda()

        # run prediction without back-propagation
        with torch.no_grad():
            output = engine(img)
            test_loss = engine.criterion(output, label)

        # compute the number of correct prediction
        pred = torch.argmax(output, dim=-1)
```

## **Future Works**

This RFC is focusing on discussing the unified device access interface. We will gradually add CPU support into the internal componenets to make the whole functionality work by following below *TODO* list.

- [ ] Gemini
- [ ] Communication
- [ ] nn/kernel
- [ ] autochunk/autoparallel
- [ ] zero
- [ ] tensor
