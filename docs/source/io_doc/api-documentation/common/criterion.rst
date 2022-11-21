Criterion
=========================================================

.. py:module:: neural_compressor.experimental.common.criterion

Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   neural_compressor.experimental.common.criterion.PyTorchKnowledgeDistillationLoss
   neural_compressor.experimental.common.criterion.PyTorchKnowledgeDistillationLossWrapper
   neural_compressor.experimental.common.criterion.TensorflowKnowledgeDistillationLoss
   neural_compressor.experimental.common.criterion.TensorflowKnowledgeDistillationLossWrapper
   neural_compressor.experimental.common.criterion.TensorflowKnowledgeDistillationLossExternal


.. py:class:: PyTorchKnowledgeDistillationLoss(temperature=1.0, loss_types=['CE', 'CE'], loss_weights=[0.5, 0.5], student_model=None, teacher_model=None)

   Bases: :py:obj:`KnowledgeDistillationLoss`

   The PyTorchKnowledgeDistillationLoss class inherits from KnowledgeDistillationLoss.

   .. py:method:: SoftCrossEntropy(logits, targets)

      Return SoftCrossEntropy.

      :param logits: output logits
      :type logits: tensor
      :param targets: ground truth label
      :type targets: tensor

      :returns: SoftCrossEntropy
      :rtype: tensor


   .. py:method:: KullbackLeiblerDivergence(logits, targets)

      Return KullbackLeiblerDivergence.

      :param logits: output logits
      :type logits: tensor
      :param targets: ground truth label
      :type targets: tensor

      :returns: KullbackLeiblerDivergence
      :rtype: tensor


   .. py:method:: teacher_model_forward(input, teacher_model=None, device=None)

      Teacher model forward.

      :param input: input data
      :type input: tensor
      :param teacher_model: teacher model. Defaults to None.
      :type teacher_model: torch.nn.model, optional
      :param device: device. Defaults to None.
      :type device: torch.device, optional

      :returns: output
      :rtype: tensor


   .. py:method:: teacher_student_loss_cal(student_outputs, teacher_outputs)

      Calculate loss between student model and teacher model.

      :param student_outputs: student outputs
      :type student_outputs: tensor
      :param teacher_outputs: teacher outputs
      :type teacher_outputs: tensor

      :returns: loss
      :rtype: tensor


   .. py:method:: student_targets_loss_cal(student_outputs, targets)

      Calculate loss of student model.

      :param student_outputs: student outputs
      :type student_outputs: tensor
      :param targets: groud truth label
      :type targets: tensor

      :returns: loss
      :rtype: tensor



.. py:class:: PyTorchKnowledgeDistillationLossWrapper(param_dict)

   Bases: :py:obj:`object`

   PyTorchKnowledgeDistillationLossWrapper wraps PyTorchKnowledgeDistillationLoss.

   .. py:method:: _param_check()


   .. py:method:: __call__(**kwargs)

      Return PyTorchKnowledgeDistillationLoss, param dict.

      :returns: PyTorchKnowledgeDistillationLoss
                param dict (dict): param dict
      :rtype: PyTorchKnowledgeDistillationLoss (class)



.. py:class:: TensorflowKnowledgeDistillationLoss(temperature=1.0, loss_types=['CE', 'CE'], loss_weights=[0.5, 0.5], student_model=None, teacher_model=None)

   Bases: :py:obj:`KnowledgeDistillationLoss`

   The TensorflowKnowledgeDistillationLoss class inherits from KnowledgeDistillationLoss.

   .. py:method:: SoftCrossEntropy(targets, logits)

      Return SoftCrossEntropy.

      :param logits: output logits
      :type logits: tensor
      :param targets: ground truth label
      :type targets: tensor

      :returns: SoftCrossEntropy
      :rtype: tensor


   .. py:method:: teacher_model_forward(input, teacher_model=None)

      Teacher model forward.

      :param input: input data
      :type input: tensor
      :param teacher_model: teacher model. Defaults to None.
      :type teacher_model: optional
      :param device: device. Defaults to None.
      :type device: torch.device, optional

      :returns: output
      :rtype: tensor


   .. py:method:: teacher_student_loss_cal(student_outputs, teacher_outputs)

      Calculate loss between student model and teacher model.

      :param student_outputs: student outputs
      :type student_outputs: tensor
      :param teacher_outputs: teacher outputs
      :type teacher_outputs: tensor

      :returns: loss
      :rtype: tensor


   .. py:method:: student_targets_loss_cal(student_outputs, targets)

      Calculate loss of student model.

      :param student_outputs: student outputs
      :type student_outputs: tensor
      :param targets: groud truth label
      :type targets: tensor

      :returns: loss
      :rtype: tensor


   .. py:method:: __call__(student_outputs, targets)

      Return loss of student model.

      :param student_outputs: student outputs
      :type student_outputs: tensor
      :param targets: groud truth label
      :type targets: tensor

      :returns: loss
      :rtype: tensor



.. py:class:: TensorflowKnowledgeDistillationLossWrapper(param_dict)

   Bases: :py:obj:`object`

   TensorflowKnowledgeDistillationLossWrapper wraps TensorflowKnowledgeDistillationLoss.

   .. py:method:: _param_check()


   .. py:method:: __call__(**kwargs)

      Return TensorflowKnowledgeDistillationLoss, param dict.

      :returns: TensorflowKnowledgeDistillationLoss
                param dict (dict): param dict
      :rtype: class



.. py:class:: TensorflowKnowledgeDistillationLossExternal(temperature=1.0, loss_types=['CE', 'CE'], loss_weights=[0.5, 0.5], student_model=None, teacher_model=None)

   Bases: :py:obj:`KnowledgeDistillationLoss`

   TensorflowKnowledgeDistillationLossExternal inherits from KnowledgeDistillationLoss.
