Metric
======================================================

.. py:module:: neural_compressor.experimental.metric.metric

Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   neural_compressor.experimental.metric.metric.BaseMetric
   neural_compressor.experimental.metric.metric.F1
   neural_compressor.experimental.metric.metric.Accuracy
   neural_compressor.experimental.metric.metric.PyTorchLoss
   neural_compressor.experimental.metric.metric.Loss
   neural_compressor.experimental.metric.metric.MAE
   neural_compressor.experimental.metric.metric.RMSE
   neural_compressor.experimental.metric.metric.MSE
   neural_compressor.experimental.metric.metric.TensorflowTopK
   neural_compressor.experimental.metric.metric.GeneralTopK
   neural_compressor.experimental.metric.metric.COCOmAPv2
   neural_compressor.experimental.metric.metric.TensorflowMAP
   neural_compressor.experimental.metric.metric.TensorflowCOCOMAP
   neural_compressor.experimental.metric.metric.TensorflowVOCMAP
   neural_compressor.experimental.metric.metric.SquadF1
   neural_compressor.experimental.metric.metric.mIOU
   neural_compressor.experimental.metric.metric.ONNXRTGLUE
   neural_compressor.experimental.metric.metric.ROC





.. py:class:: BaseMetric(metric, single_output=False, hvd=None)

   Bases: :py:obj:`object`

   The base class of Metric.

   .. py:method:: __call__(*args, **kwargs)

      Evaluate the model predictions, and the reference.

      :returns: The class itself.


   .. py:method:: update(preds, labels=None, sample_weight=None)
      :abstractmethod:

      Update the state that need to be evaluated.

      :param preds: The prediction result.
      :param labels: The reference. Defaults to None.
      :param sample_weight: The sampling weight. Defaults to None.

      :raises NotImplementedError: The method should be implemented by subclass.


   .. py:method:: reset()
      :abstractmethod:

      Clear the predictions and labels.

      :raises NotImplementedError: The method should be implemented by subclass.


   .. py:method:: result()
      :abstractmethod:

      Evaluate the difference between predictions and labels.

      :raises NotImplementedError: The method should be implemented by subclass.


   .. py:method:: metric()
      :property:

      Return its metric class.

      :returns: The metric class.


   .. py:method:: hvd()
      :property:

      Return its hvd class.

      :returns: The hvd class.



.. py:class:: F1

   Bases: :py:obj:`BaseMetric`

   F1 score of a binary classification problem.

   The F1 score is the harmonic mean of the precision and recall.
   It can be computed with the equation:
   F1 = 2 * (precision * recall) / (precision + recall)

   .. py:method:: update(preds, labels)

      Add the predictions and labels.

      :param preds: The predictions.
      :param labels: The labels corresponding to the predictions.


   .. py:method:: reset()

      Clear the predictions and labels.


   .. py:method:: result()

      Compute the F1 score.








.. py:class:: Accuracy

   Bases: :py:obj:`BaseMetric`

   The Accuracy for the classification tasks.

   The accuracy score is the proportion of the total number of predictions
   that were correct classified.

   .. attribute:: pred_list

      List of prediction to score.

   .. attribute:: label_list

      List of labels to score.

   .. attribute:: sample

      The total number of samples.

   .. py:method:: update(preds, labels, sample_weight=None)

      Add the predictions and labels.

      :param preds: The predictions.
      :param labels: The labels corresponding to the predictions.
      :param sample_weight: The sample weight.


   .. py:method:: reset()

      Clear the predictions and labels.


   .. py:method:: result()

      Compute the accuracy.



.. py:class:: PyTorchLoss

   A dummy PyTorch Metric.

   A dummy metric that computes the average of predictions and prints it directly.

   .. py:method:: reset()

      Reset the number of samples and total cases to zero.


   .. py:method:: update(output)

      Add the predictions.

      :param output: The predictions.


   .. py:method:: compute()

      Compute the  average of predictions.

      :raises ValueError: There must have at least one example.

      :returns: The dummy loss.



.. py:class:: Loss

   Bases: :py:obj:`BaseMetric`

   A dummy Metric.

   A dummy metric that computes the average of predictions and prints it directly.

   .. attribute:: sample

      The number of samples.

   .. attribute:: sum

      The sum of prediction.

   .. py:method:: update(preds, labels, sample_weight=None)

      Add the predictions and labels.

      :param preds: The predictions.
      :param labels: The labels corresponding to the predictions.
      :param sample_weight: The sample weight.


   .. py:method:: reset()

      Reset the number of samples and total cases to zero.


   .. py:method:: result()

      Compute the  average of predictions.

      :returns: The dummy loss.



.. py:class:: MAE(compare_label=True)

   Bases: :py:obj:`BaseMetric`

   Computes Mean Absolute Error (MAE) loss.

   Mean Absolute Error (MAE) is the mean of the magnitude of
   difference between the predicted and actual numeric values.

   .. attribute:: pred_list

      List of prediction to score.

   .. attribute:: label_list

      List of references corresponding to the prediction result.

   .. attribute:: compare_label

      Whether to compare label. False if there are no
      labels and will use FP32 preds as labels.

      :type: bool

   .. py:method:: update(preds, labels, sample_weight=None)

      Add the predictions and labels.

      :param preds: The predictions.
      :param labels: The labels corresponding to the predictions.
      :param sample_weight: The sample weight.


   .. py:method:: reset()

      Clear the predictions and labels.


   .. py:method:: result()

      Compute the MAE score.

      :returns: The MAE score.



.. py:class:: RMSE(compare_label=True)

   Bases: :py:obj:`BaseMetric`

   Computes Root Mean Squared Error (RMSE) loss.

   .. attribute:: mse

      The instance of MSE Metric.

   .. py:method:: update(preds, labels, sample_weight=None)

      Add the predictions and labels.

      :param preds: The predictions.
      :param labels: The labels corresponding to the predictions.
      :param sample_weight: The sample weight.


   .. py:method:: reset()

      Clear the predictions and labels.


   .. py:method:: result()

      Compute the RMSE score.

      :returns: The RMSE score.



.. py:class:: MSE(compare_label=True)

   Bases: :py:obj:`BaseMetric`

   Computes Mean Squared Error (MSE) loss.

   Mean Squared Error(MSE) represents the average of the squares of errors.
   For example, the average squared difference between the estimated values
   and the actual values.

   .. attribute:: pred_list

      List of prediction to score.

   .. attribute:: label_list

      List of references corresponding to the prediction result.

   .. attribute:: compare_label

      Whether to compare label. False if there are no labels
      and will use FP32 preds as labels.

      :type: bool

   .. py:method:: update(preds, labels, sample_weight=None)

      Add the predictions and labels.

      :param preds: The predictions.
      :param labels: The labels corresponding to the predictions.
      :param sample_weight: The sample weight.


   .. py:method:: reset()

      Clear the predictions and labels.


   .. py:method:: result()

      Compute the MSE score.

      :returns: The MSE score.



.. py:class:: TensorflowTopK(k=1)

   Bases: :py:obj:`BaseMetric`

   Compute Top-k Accuracy classification score for Tensorflow model.

   This metric computes the number of times where the correct label is among
   the top k labels predicted.

   .. attribute:: k

      The number of most likely outcomes considered to find the correct label.

      :type: int

   .. attribute:: num_correct

      The number of predictions that were correct classified.

   .. attribute:: num_sample

      The total number of predictions.

   .. py:method:: update(preds, labels, sample_weight=None)

      Add the predictions and labels.

      :param preds: The predictions.
      :param labels: The labels corresponding to the predictions.
      :param sample_weight: The sample weight.


   .. py:method:: reset()

      Reset the number of samples and correct predictions.


   .. py:method:: result()

      Compute the top-k score.

      :returns: The top-k score.



.. py:class:: GeneralTopK(k=1)

   Bases: :py:obj:`BaseMetric`

   Compute Top-k Accuracy classification score.

   This metric computes the number of times where the correct label is among
   the top k labels predicted.

   .. attribute:: k

      The number of most likely outcomes considered to find the correct label.

      :type: int

   .. attribute:: num_correct

      The number of predictions that were correct classified.

   .. attribute:: num_sample

      The total number of predictions.

   .. py:method:: update(preds, labels, sample_weight=None)

      Add the predictions and labels.

      :param preds: The predictions.
      :param labels: The labels corresponding to the predictions.
      :param sample_weight: The sample weight.


   .. py:method:: reset()

      Reset the number of samples and correct predictions.


   .. py:method:: result()

      Compute the top-k score.

      :returns: The top-k score.



.. py:class:: COCOmAPv2(anno_path=None, iou_thrs='0.5:0.05:0.95', map_points=101, map_key='DetectionBoxes_Precision/mAP', output_index_mapping={'num_detections': -1, 'boxes': 0, 'scores': 1, 'classes': 2})

   Bases: :py:obj:`BaseMetric`

   Compute mean average precision of the detection task.

   .. py:method:: update(predicts, labels, sample_weight=None)

      Add the predictions and labels.

      :param predicts: The predictions.
      :param labels: The labels corresponding to the predictions.
      :param sample_weight: The sample weight. Defaults to None.


   .. py:method:: reset()

      Reset the prediction and labels.


   .. py:method:: result()

      Compute mean average precision.

      :returns: The mean average precision score.



.. py:class:: TensorflowMAP(anno_path=None, iou_thrs=0.5, map_points=0, map_key='DetectionBoxes_Precision/mAP')

   Bases: :py:obj:`BaseMetric`

   Computes mean average precision.

   .. py:method:: update(predicts, labels, sample_weight=None)

      Add the predictions and labels.

      :param predicts: The predictions.
      :param labels: The labels corresponding to the predictions.
      :param sample_weight: The sample weight.


   .. py:method:: reset()

      Reset the prediction and labels.


   .. py:method:: result()

      Compute mean average precision.

      :returns: The mean average precision score.



.. py:class:: TensorflowCOCOMAP(anno_path=None, iou_thrs=None, map_points=None, map_key='DetectionBoxes_Precision/mAP')

   Bases: :py:obj:`TensorflowMAP`

   Computes mean average precision using algorithm in COCO.


.. py:class:: TensorflowVOCMAP(anno_path=None, iou_thrs=None, map_points=None, map_key='DetectionBoxes_Precision/mAP')

   Bases: :py:obj:`TensorflowMAP`

   Computes mean average precision using algorithm in VOC.


.. py:class:: SquadF1

   Bases: :py:obj:`BaseMetric`

   Evaluate for v1.1 of the SQuAD dataset.

   .. py:method:: update(preds, labels, sample_weight=None)

      Add the predictions and labels.

      :param preds: The predictions.
      :param labels: The labels corresponding to the predictions.
      :param sample_weight: The sample weight.


   .. py:method:: reset()

      Reset the score list.


   .. py:method:: result()

      Compute F1 score.



.. py:class:: mIOU(num_classes=21)

   Bases: :py:obj:`BaseMetric`

   Compute the mean IOU(Intersection over Union) score.

   .. py:method:: update(preds, labels)

      Add the predictions and labels.

      :param preds: The predictions.
      :param labels: The labels corresponding to the predictions.


   .. py:method:: reset()

      Reset the hist.


   .. py:method:: result()

      Compute mean IOU.

      :returns: The mean IOU score.



.. py:class:: ONNXRTGLUE(task='mrpc')

   Bases: :py:obj:`BaseMetric`

   Compute the GLUE score.

   .. py:method:: update(preds, labels)

      Add the predictions and labels.

      :param preds: The predictions.
      :param labels: The labels corresponding to the predictions.


   .. py:method:: reset()

      Reset the prediction and labels.


   .. py:method:: result()

      Compute the GLUE score.



.. py:class:: ROC(task='dlrm')

   Bases: :py:obj:`BaseMetric`

   Computes ROC score.

   .. py:method:: update(preds, labels)

      Add the predictions and labels.

      :param preds: The predictions.
      :param labels: The labels corresponding to the predictions.


   .. py:method:: reset()

      Reset the prediction and labels.


   .. py:method:: result()

      Compute the ROC score.


