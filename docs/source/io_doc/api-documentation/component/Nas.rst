NAS 
============================================

.. py:module:: neural_compressor.experimental.nas


.. autoapisummary::

   neural_compressor.experimental.nas.basic_nas
   neural_compressor.experimental.nas.dynas


Package Contents
----------------

Classes
~~~~~~~

.. autoapisummary::

   neural_compressor.experimental.nas.BasicNAS
   neural_compressor.experimental.nas.DyNAS
   neural_compressor.experimental.nas.NAS




.. py:class:: BasicNAS(conf_fname_or_obj, search_space=None, model_builder=None)

   Bases: :py:obj:`neural_compressor.experimental.nas.nas.NASBase`, :py:obj:`neural_compressor.experimental.component.Component`

   :param conf_fname: The path to the YAML configuration file.
   :type conf_fname: string
   :param search_space: A dictionary for defining the search space.
   :type search_space: dict
   :param model_builder: A function to build model instance with the specified
                         model architecture parameters.
   :type model_builder: function obj

   .. py:method:: execute()

      Initialize the dataloader and train/eval functions from yaml config.
      Component base class provides default function to initialize dataloaders and functions
      from user config. And for derived classes(Pruning, Quantization, etc.), an override
      function is required.


   .. py:method:: estimate(model)

      Estimate performance of the model.
         Depends on specific NAS algorithm. Here we use train and evaluate.

      :returns: Evaluated metrics of the model.


   .. py:method:: init_by_cfg(conf_fname_or_obj)


   .. py:method:: pre_process()

      Initialize the dataloader and train/eval functions from yaml config.
      Component base class provides default function to initialize dataloaders and functions
      from user config. And for derived classes(Pruning, Quantization, etc.), an override
      function is required.


   .. py:method:: __repr__()

      Return repr(self).



.. py:class:: DyNAS(conf_fname_or_obj)

   Bases: :py:obj:`neural_compressor.experimental.nas.nas.NASBase`

   :param conf_fname_or_obj: The path to the YAML configuration file or the object of NASConfig.
   :type conf_fname_or_obj: string or obj

   .. py:method:: estimate(individual)

      Estimate performance of the model. Depends on specific NAS algorithm.

      :returns: Evaluated metrics of the model.


   .. py:method:: init_for_search()


   .. py:method:: search()

      NAS search process.

      :returns: Best model architecture found in search process.


   .. py:method:: select_model_arch()

      Propose architecture of the model based on search algorithm for next search iteration.

      :returns: Model architecture description.


   .. py:method:: create_acc_predictor()


   .. py:method:: create_macs_predictor()


   .. py:method:: create_latency_predictor()


   .. py:method:: init_cfg(conf_fname_or_obj)



.. py:class:: NAS

   Bases: :py:obj:`object`

