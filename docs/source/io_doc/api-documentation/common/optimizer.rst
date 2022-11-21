Optimizer
=========================================================

.. py:module:: neural_compressor.experimental.common.optimizer

.. autoapi-nested-parse::

   Intel Neural Compressor built-in Optimizers on multiple framework backends.


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   neural_compressor.experimental.common.optimizer.TensorflowOptimizers
   neural_compressor.experimental.common.optimizer.PyTorchOptimizers
   neural_compressor.experimental.common.optimizer.Optimizers
   neural_compressor.experimental.common.optimizer.TensorFlowSGD
   neural_compressor.experimental.common.optimizer.TensorFlowAdamW
   neural_compressor.experimental.common.optimizer.TensorFlowAdam
   neural_compressor.experimental.common.optimizer.PyTorchSGD



Functions
~~~~~~~~~

.. autoapisummary::

   neural_compressor.experimental.common.optimizer.optimizer_registry



Attributes
~~~~~~~~~~

.. autoapisummary::

   neural_compressor.experimental.common.optimizer.torch
   neural_compressor.experimental.common.optimizer.tf
   neural_compressor.experimental.common.optimizer.tfa
   neural_compressor.experimental.common.optimizer.framework_optimizers
   neural_compressor.experimental.common.optimizer.TENSORFLOW_OPTIMIZERS
   neural_compressor.experimental.common.optimizer.PYTORCH_OPTIMIZERS
   neural_compressor.experimental.common.optimizer.registry_optimizers


.. py:data:: torch
   

   

.. py:data:: tf
   

   

.. py:data:: tfa
   

   

.. py:class:: TensorflowOptimizers

   Bases: :py:obj:`object`

   Class to get all registered TensorFlow Optimizers once only.


.. py:class:: PyTorchOptimizers

   Bases: :py:obj:`object`

   Class to get all registered PyTorch Optimizers once only.


.. py:data:: framework_optimizers
   

   

.. py:data:: TENSORFLOW_OPTIMIZERS
   

   

.. py:data:: PYTORCH_OPTIMIZERS
   

   

.. py:data:: registry_optimizers
   

   

.. py:class:: Optimizers(framework)

   Bases: :py:obj:`object`

   Main entry to get the specific type of optimizer.

   .. py:method:: __getitem__(optimizer_type)

      Return the specific type of optimizer object according to the given optimizer_type.


   .. py:method:: register(name, optimizer_cls)

      Allow registration of non-built-in optimizers.



.. py:function:: optimizer_registry(optimizer_type, framework)

   Class decorator used to register all Optimizer subclasses.

      Cross framework optimizer is supported by add param as framework='tensorflow, pytorch'

   :param optimizer_type: The string of supported criterion.
   :type optimizer_type: str
   :param framework: The string of supported framework.
   :type framework: str

   :returns: The class of register.
   :rtype: cls


.. py:class:: TensorFlowSGD(param_dict)

   Bases: :py:obj:`object`

   TensorFlow keras SGD optimizer.

   :param param_dict: The dict of parameters setting by user for SGD optimizer
   :type param_dict: dict

   .. py:method:: _mapping()


   .. py:method:: __call__(**kwargs)

      Call `TensorFlowSGD` object.



.. py:class:: TensorFlowAdamW(param_dict)

   Bases: :py:obj:`object`

   tensorflow_addons AdamW optimizer.

   :param param_dict: The dict of parameters setting by user for AdamW optimizer
   :type param_dict: dict

   .. py:method:: _mapping()


   .. py:method:: __call__(**kwargs)

      Call `TensorFlowAdamW` object.



.. py:class:: TensorFlowAdam(param_dict)

   Bases: :py:obj:`object`

   tensorflow Adam optimizer.

   :param param_dict: The dict of parameters setting by user for Adam optimizer
   :type param_dict: dict

   .. py:method:: _mapping()


   .. py:method:: __call__(**kwargs)

      Call `TensorFlowAdam` object.



.. py:class:: PyTorchSGD(param_dict)

   Bases: :py:obj:`object`

   PyTorch SGD optimizer.

   :param param_dict: The dict of parameters setting by user for SGD optimizer
   :type param_dict: dict

   .. py:method:: _mapping()


   .. py:method:: __call__(**kwargs)

      Call `PyTorchSGD` object.


